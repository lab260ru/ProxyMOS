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
import utmosv2

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
        # create_model returns model instance
        self.model = utmosv2.create_model(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ UTMOSv2 model loaded successfully")
        print(f"⚙️  Device: {self.device}")
        print(f"📊 Output: MOS score (1–5)\n")

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Convert stereo → mono if needed, resample to 16kHz, normalize.
        Return a torch.Tensor of shape (batch_size, num_samples) on self.device.
        """
        # assume audio_batch shape: (batch_size, channels, time) or (batch_size, 1, time)
        if audio_batch.ndim == 3:
            # batch, channel, time
            if audio_batch.shape[1] > 1:
                # average channels
                audio_batch = audio_batch.mean(dim=1, keepdim=False)  # (batch, time)
            else:
                audio_batch = audio_batch.squeeze(1)  # (batch, time)
        elif audio_batch.ndim == 2:
            # (batch, time) already mono
            pass
        else:
            raise ValueError(f"Unexpected audio_batch shape: {audio_batch.shape}")

        target_sr = 16000
        if sample_rate != target_sr:
            if not hasattr(self, "_resampler_cache"):
                self._resampler_cache = {}
            if sample_rate not in self._resampler_cache:
                self._resampler_cache[sample_rate] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=target_sr
                ).to(self.device)
            audio_batch = self._resampler_cache[sample_rate](audio_batch.to(self.device))
        else:
            audio_batch = audio_batch.to(self.device)

        # Normalize each sample independently
        max_val = audio_batch.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9)
        audio_batch = audio_batch / max_val

        return audio_batch

    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run inference on batch of audio tensors.
        Returns:
            {"mos": torch.Tensor([batch_size])}
        """
        batch_size = audio.shape[0]
        mos_scores: List[float] = []
        # We'll save each tensor to a temporary wav and call model.predict
        for i in range(batch_size):
            sample = audio[i].cpu().numpy()
            # write temporary wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=".") as tmpfile:
                tmpfile_path = tmpfile.name
                sf.write(tmpfile_path, sample, 16000, format="WAV")
            try:
                # Official API: model.predict(input_path=...) -> mos (float or list)
                mos = self.model.predict(input_path=tmpfile_path, num_workers=0)
            finally:
                # cleanup
                try:
                    os.unlink(tmpfile_path)
                except Exception:
                    pass

            # unify output type
            if isinstance(mos, (list, tuple)):
                mos_val = float(mos[0])
            elif isinstance(mos, torch.Tensor):
                mos_val = float(mos.item())
            else:
                mos_val = float(mos)
            mos_scores.append(mos_val)

        # convert to tensor on device
        mos_tensor = torch.tensor(mos_scores, dtype=torch.float32, device=self.device)
        return {"mos": mos_tensor}

    def postprocess(self, output: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
        """
        Convert model output to list of dicts: [{"mos": value}, ...]
        """
        mos_values = output["mos"].cpu().numpy().tolist()
        results = [{"mos": float(v)} for v in mos_values]
        return results

    @property
    def sample_rate(self) -> int:
        """Expected sample rate for the model."""
        return 16000
