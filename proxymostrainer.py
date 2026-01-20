import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import logging


class ProxyMOSTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
        use_scheduler: bool = True,
        save_dir: str = "./checkpoints",
        log_interval: int = 50,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.log_interval = log_interval

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=True,  
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.scheduler = None
        if use_scheduler:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=len(self.train_loader) * 100,  
                eta_min=lr * 0.01,
            )

        self.criterion = nn.MSELoss()
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))
        self.max_grad_norm = max_grad_norm

    def _step(self, batch, train: bool):
        wave = batch["waveform"].to(self.device, non_blocking=True)
        mos = batch["mos"].to(self.device, non_blocking=True)


        with autocast(device_type=self.device.type):
            pred = self.model(wave).squeeze(-1)
            loss = self.criterion(pred, mos)


        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.max_grad_norm
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler:
                self.scheduler.step()
            
            return loss.item(), pred.detach(), mos.detach(), grad_norm.item()
        
        return loss.item(), pred.detach(), mos.detach()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        losses = []
        preds, gts = [], []

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            loss, p, g = self._step(batch, train=False)
            losses.append(loss)
            preds.append(p)
            gts.append(g)

        preds = torch.cat(preds)
        gts = torch.cat(gts)

        mse = nn.functional.mse_loss(preds, gts)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(preds - gts))
        
        mean_pred = preds.mean()
        mean_gt = gts.mean()
        num = ((preds - mean_pred) * (gts - mean_gt)).sum()
        den = torch.sqrt(((preds - mean_pred) ** 2).sum() * ((gts - mean_gt) ** 2).sum())
        corr = num / (den + 1e-8)

        return {
            'loss': float(torch.mean(torch.tensor(losses))),
            'rmse': float(rmse),
            'mae': float(mae),
            'corr': float(corr),
        }

    def train(self, epochs: int, early_stop_patience: Optional[int] = 100):
        best_rmse = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            epoch_grad_norms = []

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for i, batch in enumerate(pbar):
                loss, _, _, grad_norm = self._step(batch, train=True)
                epoch_losses.append(loss)
                epoch_grad_norms.append(grad_norm)

                if i % self.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'grad': f'{grad_norm:.3f}'
                    })

            val_metrics = self.validate()
            train_loss = sum(epoch_losses) / len(epoch_losses)
            avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)

            self.logger.info(
                f"[Epoch {epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f} | "
                f"Val Corr: {val_metrics['corr']:.4f} | "
                f"Avg Grad: {avg_grad_norm:.3f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_metrics['rmse'] < best_rmse:
                best_rmse = val_metrics['rmse']
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'scaler_state_dict': self.scaler.state_dict(),
                    'best_rmse': best_rmse,
                    'val_metrics': val_metrics,
                }
                
                torch.save(checkpoint, self.save_dir / "best_model.pt")
                self.logger.info(f"✓ Saved best model (RMSE: {best_rmse:.4f})")
            else:
                patience_counter += 1


            if early_stop_patience and patience_counter >= early_stop_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Save checkpoint every N epochs
            if (epoch + 1) % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    self.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
                )

        self.logger.info(f"Training completed. Best RMSE: {best_rmse:.4f}")
        return best_rmse

