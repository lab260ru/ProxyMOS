import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import  numpy as  np
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import  spearmanr, pearsonr 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fairseq2.nn import BatchLayout


class AttentiveStatsPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, T, D]
        padding_mask: torch.Tensor | None = None  # [B, T], True = pad
    ) -> torch.Tensor:
        """
        Returns: [B, 2D]
        """
        # scores: [B, T]
        scores = self.att(x).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  
        
        mean = torch.sum(weights * x, dim=1)
        var = torch.sum(weights * (x - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var + 1e-6)  
        
        return torch.cat([mean, std], dim=-1)



class OmniMOS(nn.Module):
    """
    Mean Opinion Score (MOS) prediction model built on top of a Wav2Vec2-style encoder.
 
    Args:
        config_omni: Model configuration object (passed to encoder).
        encoder (nn.Module): Feature extraction encoder (e.g. Wav2Vec2).
        hidden_dim (int): Hidden dimensionality. Default: 1024.
        attentive_pooling (bool): Use attentive stats pooling instead of mean pooling.
    """
 
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 1024,
        attentive_pooling: bool = True,
    ):
        super().__init__()
 
        self.encoder = encoder
 
        dim = hidden_dim
 
        if attentive_pooling:
            self.pool = AttentiveStatsPooling(dim)
            pooled_dim = dim * 2
        else:
            self.pool = None
            pooled_dim = dim
 
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
 
    @torch.inference_mode()
    def inference(self, wave: torch.Tensor) -> torch.Tensor:
        """Run forward pass in eval mode without gradient tracking."""
        self.eval()
        return self.forward(wave)
 
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wave (torch.Tensor): Waveform tensor of shape [B, T] or [B, 1, T].
 
        Returns:
            torch.Tensor: MOS scores of shape [B].
        """
        wave = wave.float()
 
        # Normalize input shape to [B, T]
        if wave.dim() == 3 and wave.shape[1] == 1:
            wave = wave.squeeze(1)
        if wave.dim() == 3:
            wave = wave.mean(dim=1)
 
        B, T = wave.shape
 
        seqs_layout = BatchLayout(
            shape=(B, T),
            seq_lens=[T] * B,
            packed=False,
            device=wave.device,
        )
 
        # Extract features from the encoder
        features = self.encoder.extract_features(wave, seqs_layout)
 
        # Handle various return types from extract_features
        if hasattr(features, "seqs"):
            feats = features.seqs              # [B, T, D]
        elif hasattr(features, "encoder_output"):
            feats = features.encoder_output
        elif isinstance(features, tuple):
            feats = features[0]
        else:
            feats = features
 
        # Pool across time dimension
        if self.pool is not None:
            pooled = self.pool(feats, None)    # [B, 2D]
        else:
            pooled = feats.mean(dim=1)         # [B, D]
 
        return self.head(pooled).squeeze(-1)   



    

class ProxyMOSTrainer:
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
        save_dir: str = "./checkpoints",
    ):
        self.accelerator = accelerator
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_grad_norm = max_grad_norm
        self.history = []
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.patience = 0
        self.best_score = -float('inf')
        
        encoder_params = list(model.encoder.parameters())
        pool_params = list(model.pool.parameters()) if model.pool else []
        head_params = list(model.head.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr * 0.01},  
            {"params": pool_params, "lr": lr},
            {"params": head_params, "lr": lr},
        ], weight_decay=weight_decay)
        
        self.criterion = nn.MSELoss()
        
        # Accelerate prepare
        self.model, self.optimizer, self.train_loader, self.val_loader = (
            accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader,
            )
        )


    def _validate_and_save(self, epoch: int, tag: str):
        self.model.eval()
        metrics = self.validate()
        self.model.train()
        
        self.accelerator.wait_for_everyone()
        
        should_save = False
        if self.accelerator.is_main_process and metrics is not None:
            cur_score = (metrics["corr"] + metrics["corr_per"]) / 2

            log = {
                "epoch": epoch + 1,
                "tag": tag,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "corr": metrics["corr"],
                "score": cur_score,
            }
            self.history.append(log)
            print(
                f"[Epoch {epoch+1} | {tag}] "
                f"RMSE={metrics['rmse']:.4f} | "
                f"MAE={metrics['mae']:.4f} | "
                f"Corr={metrics['corr']:.4f} | "
                f"Score={cur_score:.4f}"

            )
            
            if cur_score > self.best_score:
                print("🔥 New best model (composite metric)")
                self.best_score = cur_score
                self.patience = 0
                should_save = True
            else:
                self.patience += 1
        if self.accelerator.num_processes > 1: 
            should_save = self.accelerator.reduce( 
                                                  torch.tensor(should_save, device=self.accelerator.device), 
                                                  reduction="sum" ).item() > 0
        
        if should_save:
            self.save_full(tag)
        

    def save_full(self, tag: str):
        state = self.accelerator.get_state_dict(self.model)
        
        if self.accelerator.is_main_process:
            print("💾 Saving state dict to disk...")
            torch.save(state, self.save_dir / f"best_model_{tag}.pt")
            print("✅ FULL checkpoint saved")
        
        self.accelerator.wait_for_everyone()

    def train(self, epochs: int = 50, early_stop_patience: int = 20):
        steps_per_epoch = len(self.train_loader)
        half_epoch_step = steps_per_epoch // 2
        

        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        
        warmup_epochs = 2
        warmup_steps = steps_per_epoch * warmup_epochs
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=steps_per_epoch * epochs - warmup_steps,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            [warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        for epoch in range(epochs):
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            
            self.model.train()
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                disable=not self.accelerator.is_main_process,
            )
            
            for step, batch in enumerate(pbar):
                wave = batch["waveform"]
                mos = batch["mos"]
                
                pred = self.model(wave)
                loss = self.criterion(pred, mos)
                
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                
                if self.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                
                if self.accelerator.is_main_process:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                
                if step + 1 == half_epoch_step :
                    self._validate_and_save(epoch, tag="half")
            
            self._validate_and_save(epoch, tag="full")
            
            if early_stop_patience and self.patience >= early_stop_patience:
                if self.accelerator.is_main_process:
                    print("🛑 Early stopping triggered")
                break
        
            self.save_full(tag=f"{epoch+1}")
            
        if self.accelerator.is_main_process:
            import json
            with open(self.save_dir / "training_log.json", "w") as f:
                json.dump(self.history, f, indent=2)
        
        return self.best_score

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        preds = []
        proxy_gts = []


        for batch in self.val_loader:
            wave = batch["waveform"]
            mos_proxy = batch["mos"]


            pred = self.model(wave)

            preds.append(pred)
            proxy_gts.append(mos_proxy)


        preds = torch.cat(preds)
        proxy_gts = torch.cat(proxy_gts)


        preds = self.accelerator.gather(preds).cpu().numpy()
        proxy_gts = self.accelerator.gather(proxy_gts).cpu().numpy()

        if not self.accelerator.is_main_process:
            return None

        # =========================================================
        # 1️⃣ PROXY METRICS (utterance level)
        # =========================================================
        pearson_proxy = pearsonr(preds, proxy_gts)[0]
        spearman_proxy = spearmanr(preds, proxy_gts)[0]
        rmse_proxy = np.sqrt(mean_squared_error(proxy_gts, preds))
        mae_proxy = mean_absolute_error(proxy_gts, preds)

        return {
            # proxy metrics
            "rmse": rmse_proxy,
            "mae": mae_proxy,
            "corr": spearman_proxy,
            "corr_per": pearson_proxy

        }



