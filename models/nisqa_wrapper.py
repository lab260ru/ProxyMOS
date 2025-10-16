# models/nisqa_wrapper.py
import torch
import torchaudio
import numpy as np
from typing import Dict, List
from pathlib import Path
from .base_model import BaseModelWrapper


class NISQAWrapper(BaseModelWrapper):
    """
    Wrapper for NISQA (Non-Intrusive Speech Quality Assessment) model.

    Features:
        - Predicts MOS and quality dimensions (noisiness, coloration, discontinuity, loudness)
        - Uses torchmetrics implementation with automatic weight download
        - Supports both CPU and GPU inference
        - Automatic resampling to 16 kHz
    
    Installation:
        pip install torchmetrics librosa requests
    """

    def load_model(self):
        """
        Load NISQA model using torchmetrics implementation.
        
        Config options:
            - device: "cpu" or "cuda"
        
        Note: Model weights are downloaded automatically on first use.
        """
        try:
            from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
            
            self.model = NonIntrusiveSpeechQualityAssessment(fs=16000)
            self.model.to(self.device)
            
            print(f"✅ NISQA model loaded via torchmetrics")
            print(f"⚙️ Device: {self.device}")
            print(f"📊 Output: MOS, Noisiness, Discontinuity, Coloration, Loudness\n")
            
        except ImportError as e:
            raise ImportError(
                "torchmetrics not found. Install with: pip install torchmetrics librosa requests"
            ) from e

    # ---------------------- PREPROCESS ----------------------

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Preprocess audio for NISQA model.
        
        Steps:
            1. Convert stereo to mono
            2. Resample to 16 kHz (NISQA requirement)
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

    # ---------------------- INFERENCE ----------------------

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Run NISQA model inference.
        
        Args:
            audio: Preprocessed audio [batch, samples]
            
        Returns:
            torch.Tensor: [batch, 5] predictions
                          (MOS, Noisiness, Discontinuity, Coloration, Loudness)
        """
        # Try batched inference first; fallback to per-sample if unsupported
        with torch.no_grad():
            try:
                scores = self.model(audio)  # expected [B, 5]
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)
                return scores
            except Exception:
                batch_size = audio.shape[0]
                all_scores = []
                for i in range(batch_size):
                    sample = audio[i]
                    s = self.model(sample)  # [5]
                    all_scores.append(s)
                return torch.stack(all_scores)

    # ---------------------- POSTPROCESS ----------------------

    def postprocess(self, output: torch.Tensor) -> List[Dict[str, float]]:
        """
        Convert model output to readable NISQA metrics.
        
        Args:
            output: Model predictions [batch, 5]
            
        Returns:
            List[Dict[str, float]]: NISQA quality metrics
        """
        results = []
        
        for pred in output.cpu():
            results.append({
                "mos": float(pred[0]),           # Overall MOS score (1-5)
                "noisiness": float(pred[1]),     # Noisiness dimension
                "discontinuity": float(pred[2]), # Discontinuity/distortion
                "coloration": float(pred[3]),    # Coloration (frequency balance)
                "loudness": float(pred[4]),      # Loudness appropriateness
            })
        
        return results

    # ---------------------- PROPERTY ----------------------

    @property
    def sample_rate(self) -> int:
        """NISQA v2.0 requires 16 kHz audio."""
        return 16000




