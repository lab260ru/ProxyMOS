import torch
from transformers import WavLMModel
from .base_model import BaseModelWrapper

class WavLMMOSWrapper(BaseModelWrapper):
    """
    Wrapper for WavLM for non-intrusive MOS prediction.
    Uses WavLM as feature extractor and a small regressor head.
    """

    def load_model(self):
        model_name = self.config.get("model_name", "microsoft/wavlm-base-plus-sd")
        self.device = self.config.get("device", "cpu")

        # Load WavLM as feature extractor
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Regression head for MOS (hidden_dim from WavLM is 768)
        hidden_dim = 768
        self.mos_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1)
        ).to(self.device)
        self.mos_head.eval()

        # Sample rate for resampling
        self._sample_rate = 16000

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Resample audio to 16kHz if needed
        audio_batch: [batch, time] or [batch, channels, time]
        """
        if audio_batch.ndim == 3:
            audio_batch = audio_batch.mean(dim=1)  # convert to mono

        if sample_rate != self._sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sample_rate, self._sample_rate).to(self.device)
            audio_batch = resampler(audio_batch)

        # Normalize
        max_val = audio_batch.abs().max(dim=-1, keepdim=True)[0]
        audio_batch = audio_batch / (max_val + 1e-9)

        return audio_batch

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract WavLM features, mean-pool over time, predict MOS
        audio: [batch, time]
        returns: [batch]
        """
        audio = audio.to(self.device)
        results = []

        with torch.no_grad():
            for i in range(audio.shape[0]):
                waveform = audio[i].unsqueeze(0)  # [1, time]
                features = self.model(waveform).last_hidden_state  # [1, seq_len, hidden_dim]
                pooled = features.mean(dim=1)  # [1, hidden_dim]
                mos = self.mos_head(pooled).squeeze(0)  # [1] -> scalar
                results.append(mos)

        return torch.stack(results)

    def postprocess(self, output: torch.Tensor):
        """
        Convert to list of dicts for consistency
        """
        return [{"mos": float(x)} for x in output.cpu().flatten()]

    @property
    def sample_rate(self):
        return self._sample_rate

