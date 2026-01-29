import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR

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
        x: torch.Tensor,         # [B, T, D]
        padding_mask: torch.Tensor | None = None  # [B, T], True = pad
    ) -> torch.Tensor:
        """
        Returns: [B, 2D]
        """
        # scores: [B, T]
        scores = self.att(x).squeeze(-1)

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -1e9)

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]

        mean = torch.sum(weights * x, dim=1)
        var = torch.sum(weights * (x - mean.unsqueeze(1)) ** 2, dim=1)

        return torch.cat([mean, var], dim=-1)



class OmniMOS(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 512,
        attentive_pooling: bool = True,
    ):
        super().__init__()

        self.encoder = encoder
        dim = encoder.cfg.encoder_embed_dim

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
        return self.forward(wave)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 3 and wave.shape[1] == 1:
            wave = wave.squeeze(1)

        if wave.dim() == 3:
            wave = wave.mean(dim=1)

        # wave: [B, T_wave]
        padding_mask = wave.abs().eq(0)  


        out = self.encoder(
            wave,
            padding_mask=padding_mask,
            features_only=True
        )

        feats = out["x"]                       
        feat_padding_mask = out["padding_mask"] 


        pooled = self.pool(feats, feat_padding_mask)

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
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm

        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        head_params = [p for p in model.head.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr * 0.01},
            {"params": head_params, "lr": lr},
        ], weight_decay=weight_decay)

        self.criterion = nn.MSELoss()

        self.model, self.optimizer, self.train_loader, self.val_loader = accelerator.prepare(
            model, self.optimizer, train_loader, val_loader
        )

    def train(self, epochs=50, early_stop_patience=20):
        best_rmse = float("inf")
        patience = 0
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader)*epochs)

        for epoch in range(epochs):
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            self.model.train()
            losses = []
            pbar = tqdm(self.train_loader, disable=not self.accelerator.is_main_process,
                        desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                wave = batch["waveform"]
                mos = batch["mos"]

                pred = self.model(wave)
                loss = self.criterion(pred, mos)

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)

                if self.max_grad_norm:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()
                losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_metrics = self.validate()

            if self.accelerator.is_main_process:
                print(
                    f"[Epoch {epoch+1}] TrainLoss={sum(losses)/len(losses):.4f} | "
                    f"ValRMSE={val_metrics['rmse']:.4f} | "
                    f"ValMAE={val_metrics['mae']:.4f} | Corr={val_metrics['corr']:.4f}"
                )

                if val_metrics["rmse"] < best_rmse:
                    best_rmse = val_metrics["rmse"]
                    patience = 0
                    torch.save(
                        self.accelerator.get_state_dict(self.model),
                        self.save_dir / "best_model.pt"
                    )
                else:
                    patience += 1

            if early_stop_patience and patience >= early_stop_patience:
                if self.accelerator.is_main_process:
                    print("Early stopping triggered")
                break

        return best_rmse

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        preds, gts = [], []

        for batch in self.val_loader:
            wave = batch["waveform"]
            mos = batch["mos"]
            pred = self.model(wave)
            preds.append(pred)
            gts.append(mos)

        preds = self.accelerator.gather(torch.cat(preds))
        gts = self.accelerator.gather(torch.cat(gts))

        mse = torch.mean((preds - gts) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(preds - gts))
        mp, mg = preds.mean(), gts.mean()
        corr = ((preds - mp) * (gts - mg)).sum() / (
            torch.sqrt(((preds - mp) ** 2).sum()) *
            torch.sqrt(((gts - mg) ** 2).sum()) + 1e-8
        )
        return {"rmse": rmse.item(), "mae": mae.item(), "corr": corr.item()}

