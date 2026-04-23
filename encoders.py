# encoders.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from fairseq2.nn import BatchLayout
from transformers import Wav2Vec2Model,  Wav2Vec2FeatureExtractor


class BaseEncoder(ABC, nn.Module):
    """Единый контракт для всех энкодеров."""

    @abstractmethod
    def encode(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: [B, T] → features: [B, T', D]"""
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

class FairSeq2Encoder(BaseEncoder):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def encode(self, wave: torch.Tensor) -> torch.Tensor:
        B, T = wave.shape
        
        # Приводим вход к Float32 для fairseq2 модели
        if wave.dtype == torch.bfloat16:
            wave = wave.float()
        
        layout = BatchLayout(
            shape=(B, T), seq_lens=[T] * B,
            packed=False, device=wave.device,
        )
        features = self.model.extract_features(wave, layout)

        if hasattr(features, "seqs"):
            return features.seqs
        elif hasattr(features, "encoder_output"):
            return features.encoder_output
        elif isinstance(features, tuple):
            return features[0]
        return features

    @property
    def output_dim(self) -> int:
        return 1024


# encoders.py

class HuggingFaceEncoder(BaseEncoder):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model 
        _proc = Wav2Vec2FeatureExtractor.from_pretrained(
            "ylacombe/omniASR_W2V_300M_SSL",
            token = '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'
        )
        self.do_normalize = _proc.do_normalize
        self.sample_rate = _proc.sampling_rate
    def _normalize(self, wave: torch.Tensor) -> torch.Tensor:
        """Zero-mean unit-variance per utterance — то же что процессор."""
        if not self.do_normalize:
            return wave
        mean = wave.mean(dim=-1, keepdim=True)
        std = wave.std(dim=-1, keepdim=True)
        return (wave - mean) / (std + 1e-7)
    def encode(self, wave: torch.Tensor) -> torch.Tensor:
        wave = self._normalize(wave)                          # [B, T], на GPU, bf16
        out = self.model(wave)
        return out.last_hidden_state

    @property
    def output_dim(self) -> int:
        return self.model.config.hidden_size


ENCODER_REGISTRY: dict[str, type[BaseEncoder]] = {
    "fairseq2":    FairSeq2Encoder,
    "huggingface": HuggingFaceEncoder,
}


def build_encoder(encoder_type: str, model: nn.Module) -> BaseEncoder:
    cls = ENCODER_REGISTRY.get(encoder_type)
    if cls is None:
        raise ValueError(f"Unknown encoder type '{encoder_type}'. "
                         f"Available: {list(ENCODER_REGISTRY)}")
    return cls(model)