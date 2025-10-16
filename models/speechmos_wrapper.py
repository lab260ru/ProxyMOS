import torch
import torchaudio
from typing import Dict, Any, List
from .base_model import BaseModelWrapper


class SpeechMOSWrapper(BaseModelWrapper):
    """Wrapper for SpeechMOS - neural MOS predictor from Facebook Research."""

    def load_model(self):
        """Load pretrained SpeechMOS model."""
        # Using torchaudio hub or custom implementation
        model_path = self.config.get("model_path", None)
        
        if model_path:
            # Load from checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = checkpoint['model']
        else:
            # Load from torch hub (if available)
            try:
                self.model = torch.hub.load(
                    'facebookresearch/speech-resynthesis',
                    'speechmos',
                    trust_repo=True
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load SpeechMOS model: {e}")
        
        self.model = self.model.to(self.device).eval()
        self._sample_rate = self.config.get("sample_rate", 16000)

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample and normalize audio."""
        if sample_rate != self._sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self._sample_rate
            ).to(self.device)
            audio_batch = resampler(audio_batch)
        
        # Normalize to [-1, 1]
        max_val = audio_batch.abs().max(dim=-1, keepdim=True)[0]
        audio_batch = audio_batch / (max_val + 1e-9)
        return audio_batch

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Predict MOS scores."""
        with torch.no_grad():
            mos_scores = self.model(audio.to(self.device))
        return mos_scores

    def postprocess(self, output: torch.Tensor) -> List[Dict[str, float]]:
        """Convert to dict format with MOS score (1-5 scale)."""
        return [{"mos": float(score)} for score in output.cpu().flatten()]

    @property
    def sample_rate(self) -> int:
        return self._sample_rate