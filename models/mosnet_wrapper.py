import torch
import torchaudio
import numpy as np
from typing import Dict, List
from pathlib import Path
from .base_model import BaseModelWrapper
from .mosnet import MOSNet

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
import random 
import numpy as np

def set_seed(seed: int = 42):
    """Устанавливает случайный сид для всех библиотек"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Для воспроизводимости
    torch.backends.cudnn.benchmark = False 
    
SEED = 42  
set_seed(SEED)

class MosNetWrapper(BaseModelWrapper):
    
    
    
    def load_model(self):

        model_path = self.config.get("checkpoint_path")
            
        if not model_path:
            raise ValueError("'checkpoint_path' must be provided in config")
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        device = self.config.get("device", "cpu").lower()
        
        self.device = device
        self.mos_model = MOSNet(device=device)
        self.mos_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.mos_model.eval()
        
        
    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Preprocess audio for MOSNet model.
        
        Steps:
            1. Convert stereo to mono
            2. Resample to 16 kHz (MOSNet requirement)
            3. Normalize to [-1, 1]
        
        Args:
            audio_batch: Input tensor [batch, channels, samples]
            sample_rate: Original audio sample rate
            
        Returns:
            torch.Tensor: Preprocessed audio [batch, samples]
        """
        # Convert to mono
        if audio_batch.shape[1] > 1:
            audio_batch = audio_batch.mean(dim=1, keepdim=True)
        
        # Remove channel dimension
        audio_batch = audio_batch.squeeze(1)  # [batch, samples]
        
        # Resample to 16 kHz if needed
        target_sr = 16000
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=target_sr
            ).to(self.device)
            audio_batch = resampler(audio_batch)
        
        # Normalize to [-1, 1]
        max_val = audio_batch.abs().max(dim=-1, keepdim=True)[0]
        audio_batch = audio_batch / (max_val + 1e-9)
        
        return audio_batch.to(self.device)
        
    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run MOSNet model inference.
        
        Args:
            audio: Preprocessed audio [batch, samples]
            
        Returns:
            Dict with:
                - "mos_average": [batch] tensor with average MOS per sample
                - "mos_per_frame": [batch, frames] tensor with frame-level MOS
        """
        batch_size = audio.shape[0]
        all_mos_avg = []
        all_mos_frames = []
        
        # Process each sample individually
        for i in range(batch_size):
            sample = audio[i].unsqueeze(0)  # [1, samples]
            
                # MOSNet returns (average_mos, per_frame_mos)
            mos_avg, mos_frames = self.mos_model(sample)
            
            all_mos_avg.append(mos_avg)
            all_mos_frames.append(mos_frames)
        
        return {
            "mos_average": torch.cat(all_mos_avg),      # [batch]
            "mos_per_frame": torch.cat(all_mos_frames)  # [batch, frames]
        }
        
    def postprocess(self, output: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
        """
        Convert model output to readable MOS metrics.
        
        Args:
            output: Dict with "mos_average" and "mos_per_frame" tensors
            
        Returns:
            List[Dict[str, float]]: MOS quality metrics
        """
        results = []
        
        mos_avg = output["mos_average"].cpu()
        mos_frames = output["mos_per_frame"].cpu()
        
        for i in range(len(mos_avg)):
            frame_scores = mos_frames[i].numpy()
            
            results.append({
                "mos": float(mos_avg[i]),              # Average MOS (1-5)
                "mos_std": float(frame_scores.std()),   # Temporal variability
                "mos_min": float(frame_scores.min()),   # Worst frame quality
                "mos_max": float(frame_scores.max()),   # Best frame quality
                "num_frames": len(frame_scores),        # Number of frames
            })
        
        return results


        
        
        
    @property
    def sample_rate(self) -> int:
        """MosNet always expects 16 kHz audio."""
        return 16000