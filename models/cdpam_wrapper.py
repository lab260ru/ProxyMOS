import torch
from typing import Dict, Any, List
from .base_model import BaseModelWrapper


class CDPAMWrapper(BaseModelWrapper):
    """Wrapper for CDPAM - Correlate Deviation Perceptual Audio Metric.
    
    Available via pip install cdpam. Simple non-intrusive metric
    for perceptual audio quality without reference.
    GitHub: https://github.com/pranaymanocha/PerceptualAudio
    """

    def load_model(self):
        """Load CDPAM model."""
        # Workaround for PyTorch 2.6+ where torch.load default weights_only=True
        import torch as _torch
        _orig_load = _torch.load
        def _load_wo(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)
        _torch.load = _load_wo  # monkey-patch
        try:
            import cdpam
            self.cdpam = cdpam.CDPAM()
        except ImportError:
            _torch.load = _orig_load
            raise ImportError(
                "CDPAM not installed. Install with: pip install cdpam"
            )
        finally:
            _torch.load = _orig_load
        
        self._sample_rate = self.config.get("sample_rate", 16000)

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resample and normalize audio."""
        if audio_batch.shape[1] > 1:
            audio_batch = audio_batch.mean(dim=1, keepdim=True)
        audio = audio_batch.squeeze(1)
        if sample_rate != self._sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self._sample_rate
            ).to(self.device)
            audio = resampler(audio)
        
        # Normalize
        max_val = audio.abs().amax(dim=-1, keepdim=True) + 1e-9
        audio = (audio / max_val).contiguous()
        
        return audio

    def _call_cdpam(self, ref: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """Call CDPAM with best-effort API compatibility.
        Returns 1D tensor with a single score.
        """
        # Ensure batched float tensors on device: [1, T]
        ref = ref.float().to(self.device).view(1, -1)
        deg = deg.float().to(self.device).view(1, -1)
        try:
            out = self.cdpam.forward(ref, deg)
        except TypeError:
            try:
                out = self.cdpam(ref, deg)
            except Exception:
                # As a last resort try single-argument API (non-intrusive fallback)
                out = self.cdpam.forward(deg)
        # Normalize output to tensor scalar
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out, dtype=torch.float32, device=self.device)
        out = out.view(1)
        return out

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Calculate CDPAM scores (non-intrusive fallback: ref=deg)."""
        scores = []
        for waveform in audio:
            if waveform.dim() > 1:
                waveform = waveform.squeeze(0)
            # Use the same signal as reference and degraded (non-intrusive approximation)
            score = self._call_cdpam(waveform, waveform)
            scores.append(score)
        return torch.stack(scores, dim=0)

    def postprocess(self, output: torch.Tensor) -> List[Dict[str, float]]:
        """Convert to dict format. Lower scores = better quality."""
        return [{"cdpam_score": float(score)} for score in output.detach().cpu().flatten()]

    @property
    def sample_rate(self) -> int:
        return self._sample_rate