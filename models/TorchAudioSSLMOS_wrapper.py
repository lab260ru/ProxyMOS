import torch
from typing import Dict, Any, List
from .base_model import BaseModelWrapper

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


class TorchAudioSSLMOSWrapper(BaseModelWrapper):
    """Wrapper for SSL-based MOS models available in TorchAudio/torch.hub.
    
    Uses self-supervised learning models (Wav2Vec2, HuBERT) with simple
    fine-tuning heads. Easy to load without custom architectures.
    """

    def load_model(self):
        """Load SSL model with MOS head from torch.hub or TorchAudio."""
        import torchaudio
        
        # Use pretrained SSL models available in torch/torchaudio
        ssl_model = self.config.get("ssl_model", "wav2vec2")
        
        
        if ssl_model == "wav2vec2":
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.ssl_encoder = bundle.get_model().to(self.device)
        elif ssl_model == "hubert":
            bundle = torchaudio.pipelines.HUBERT_BASE
            self.ssl_encoder = bundle.get_model().to(self.device)
        else:
            raise ValueError(f"Unknown SSL model: {ssl_model}")
        
        self.ssl_encoder.eval()
        self._sample_rate = bundle.sample_rate
        
        # Simple MOS prediction head
        hidden_dim = 768  # Standard for base models
        self.mos_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        
        # Load MOS head weights if provided
        mos_head_path = self.config.get("mos_head_path", None)
        if mos_head_path:
            self.mos_head.load_state_dict(torch.load(mos_head_path, map_location=self.device))
        
        self.mos_head.eval()

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample to model's sample rate."""
        
        if audio_batch.shape[1] > 1:
            audio_batch = audio_batch.mean(dim=1, keepdim=True)
        
        # Remove channel dimension
        audio_batch = audio_batch.squeeze(1)  # [batch, samples]
        if sample_rate != self._sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self._sample_rate
            ).to(self.device)
            audio_batch = resampler(audio_batch)
        
        # Normalize
        max_val = audio_batch.abs().max(dim=-1, keepdim=True)[0]
        audio_batch = audio_batch / (max_val + 1e-9)
        
        return audio_batch

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract SSL features and predict MOS."""
        with torch.no_grad():
            # Get SSL representations
            features, _ = self.ssl_encoder.extract_features(audio.to(self.device))
            
            # Use last layer features
            last_hidden = features[-1]
            
            # Average pool over time dimension
            pooled = last_hidden.mean(dim=1)
            
            # Predict MOS (scale to 1-5)
            mos_scores = self.mos_head(pooled) * 4 + 1
        
        return mos_scores

    def postprocess(self, output: torch.Tensor) -> List[Dict[str, float]]:
        """Convert to dict format."""
        return [{"mos": float(score)} for score in output.cpu().flatten()]

    @property
    def sample_rate(self) -> int:
        return self._sample_rate