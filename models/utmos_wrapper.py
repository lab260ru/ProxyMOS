# models/utmos_wrapper.py
import torch
import torchaudio
import numpy as np
import io
import soundfile as sf
import tempfile
import os
from typing import Dict, List
from pathlib import Path
from .base_model import BaseModelWrapper
import sys
import os
import warnings
import torch
import pytorch_lightning as pl
import numpy as np
import UTMOS.lightning_module as lightning_module

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
os.environ['HYDRA_FULL_ERROR'] = '0'  # Отключаем полные ошибки Hydra
os.environ['HYDRA_DISABLE_ERRORS'] = '1'  # Отключаем ошибки Hydra

# Ленивый импорт UTMOS.lightning_module

            
                    

class UTMOSWrapper(BaseModelWrapper):
    """
    Wrapper for UTMOSv2 (UTokyo-SaruLab MOS prediction system).

    Structure:
      - load_model()
      - preprocess()
      - forward()  → returns {"mos": tensor([...])}
      - postprocess()
    """

    def load_model(self):
        """Load pretrained UTMOSv2 model."""
        device = self.config.get("device", "cpu").lower()
        self.device = device
        

        self.model =lightning_module.BaselineLightningModule.load_from_checkpoint(self.config.get("checkpoint_path", "epoch3D7459.ckpt"))
        self.model.to(self.device , dtype=torch.bfloat16).eval()

    # ---------------------- INFERENCE ----------------------
    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run UTMOS model inference.
        
        Args:
            audio: Preprocessed audio [batch, samples]
            
        Returns:
            Dict[str, torch.Tensor]: {"mos": tensor([batch_size])}
        """
        # UTMOS model expects input in format: [B, 1, T] (with channel dimension)
        # Audio comes as [B, T], so we need to add channel dimension
        if audio.dim() == 2:
            # [B, T] -> [B, 1, T]
            audio = audio.unsqueeze(1)
        
        batch_size = audio.shape[0]
        audio = audio.to(self.device, dtype=torch.bfloat16)
        
        try:
            # Create batch dictionary as expected by UTMOS model
            # Based on score.py: 'wav', 'domains', 'judge_id'
            batch = {
                'wav': audio,  # [B, 1, T]
                'domains': torch.zeros(batch_size, dtype=torch.long, device=self.device),
                'judge_id': torch.ones(batch_size, dtype=torch.long, device=self.device) * 288
            }
            
            # Model returns tensor with time dimension: [B, T, 1] or [B, T]
            output = self.model(batch)
            
            # Take mean over time dimension to get MOS score per sample
            # Output shape can be [B, T, 1] or [B, T]
            if output.dim() == 3:
                # [B, T, 1] -> [B, 1] -> [B]
                mos_scores = output.mean(dim=1).squeeze(-1)
            elif output.dim() == 2:
                # [B, T] -> [B]
                mos_scores = output.mean(dim=1)
            else:
                # Fallback if unexpected shape
                mos_scores = output.squeeze()
            
            # Ensure output is 1D tensor and convert to float32 for compatibility
            if mos_scores.dim() == 0:
                mos_scores = mos_scores.unsqueeze(0)
            
            # Convert to float32 to avoid bfloat16 issues with numpy
            mos_scores = mos_scores.float()
            
            # Apply scaling to convert from model output range to MOS range [1, 5]
            # Based on score.py: output.mean(dim=1).squeeze(1) * 2 + 3
            mos_scores = mos_scores * 2.0 + 3.0
            
            return {"mos": mos_scores}
            
        except Exception as e:
            print(f"Batch inference failed: {e}")
            # Fallback: обрабатываем по одному
            all_scores = []
            
            for i in range(batch_size):
                sample_audio = audio[i:i+1]  # [1, 1, T]
                try:
                    batch = {
                        'wav': sample_audio,
                        'domains': torch.zeros(1, dtype=torch.long, device=self.device),
                        'judge_id': torch.ones(1, dtype=torch.long, device=self.device) * 288
                    }
                    output = self.model(batch)
                    
                    # Take mean over time dimension
                    if output.dim() == 3:
                        score = output.mean(dim=1).squeeze(-1).squeeze(0)
                    elif output.dim() == 2:
                        score = output.mean(dim=1).squeeze(0)
                    else:
                        score = output.squeeze()
                    
                    # Apply scaling to convert from model output range to MOS range [1, 5]
                    score = score.float() * 2.0 + 3.0
                    
                    all_scores.append(float(score))
                except Exception as e2:
                    print(f"Single sample inference failed: {e2}")
                    all_scores.append(3.0)  # Fallback score
            
            return {"mos": torch.tensor(all_scores, device=self.device, dtype=torch.float32)}

    def postprocess(self, output: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
        """
        Convert model output to list of dicts: [{"mos": value}, ...]
        """
        # Convert bfloat16 to float32 for numpy compatibility
        mos_tensor = output["mos"].cpu()
        if mos_tensor.dtype == torch.bfloat16:
            mos_tensor = mos_tensor.float()
        mos_values = mos_tensor.numpy().tolist()
        results = [{"mos": float(v)} for v in mos_values]
        return results

    @property
    def sample_rate(self) -> int:
        """Expected sample rate for the model."""
        return 16000
