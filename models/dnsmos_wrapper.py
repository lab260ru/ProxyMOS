# models/dnsmos_wrapper.py
import torch
import torchaudio
import numpy as np
from typing import Dict, List
from pathlib import Path
from .base_model import BaseModelWrapper


class DNSMOSWrapper(BaseModelWrapper):
    """
    Wrapper for DNSMOS model (speech quality prediction).

    Features:
        - Loads ONNX DNSMOS model (CPU or GPU)
        - Converts raw audio to log-mel spectrograms
        - Runs ONNX inference and returns MOS metrics
    """

    def load_model(self):
        """
        Load the DNSMOS ONNX model based on config device settings.
        Supported devices: 'cpu', 'cuda'
        """
        import onnxruntime as ort

        model_path = self.config.get("checkpoint_path")
        if not model_path:
            raise ValueError("'checkpoint_path' must be provided in config")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Select device
        device = self.config.get("device", "cpu").lower()
        if device == "cuda" and torch.cuda.is_available():
            providers = ["CUDAExecutionProvider"]
            print("⚙️ Using GPU (CUDAExecutionProvider) for ONNX Runtime")
        else:
            providers = ["CPUExecutionProvider"]
            device = "cpu"
            print("⚙️ Using CPUExecutionProvider for ONNX Runtime")

        self.device = device
        self.model = ort.InferenceSession(str(model_path), providers=providers)
        self.input_info = self.model.get_inputs()[0]
        self.input_name = self.input_info.name

        # Precompute spectrogram transforms (CPU/GPU-agnostic; we run them on CPU by default)
        self.target_sr = 16000
        self.n_fft = 320
        self.hop_length = 160
        self.n_mels = 161
        self.target_time_frames = 901
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

        print(f"✅ DNSMOS model loaded: {model_path.name}")
        print(f"Expected input shape: {self.input_info.shape}")
        print("This model expects spectrograms, not raw waveforms!\n")

    # ---------------------- PREPROCESS ----------------------

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Convert raw waveforms into spectrograms expected by DNSMOS (batched).

        Args:
            audio_batch (torch.Tensor): [B, C, T]
            sample_rate (int): input sample rate
        Returns:
            torch.Tensor: [B, time_frames, freq_bins]
        """
        B, C, T = audio_batch.shape
        # To mono: [B, T]
        audio_mono = audio_batch.mean(dim=1)

        # Resample to target_sr if needed (batched)
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            audio_mono = resampler(audio_mono)

        # Normalize per-sample to [-1, 1]
        max_vals = audio_mono.abs().amax(dim=1, keepdim=True) + 1e-9
        audio_mono = audio_mono / max_vals

        # MelSpectrogram expects [B, T]
        mel = self.mel_spec(audio_mono)  # [B, n_mels, frames]
        mel_db = self.amp_to_db(mel)     # [B, n_mels, frames]
        mel_db = mel_db.transpose(1, 2)  # [B, frames, n_mels]

        # Pad/trim time dimension to target_time_frames
        frames = mel_db.shape[1]
        if frames > self.target_time_frames:
            mel_db = mel_db[:, :self.target_time_frames, :]
        elif frames < self.target_time_frames:
            pad = self.target_time_frames - frames
            mel_db = torch.nn.functional.pad(mel_db, (0, 0, 0, pad), value=-80.0)

        return mel_db

    # ---------------------- INFERENCE ----------------------

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Run ONNX inference on spectrograms.
        Args:
            spectrograms (torch.Tensor): [B, time_frames, freq_bins]
        Returns:
            torch.Tensor: [B, 3] (ovrl, sig, bak)
        """
        # ONNX runtime on CPU/GPU uses numpy arrays
        spectrograms_np = spectrograms.cpu().numpy().astype(np.float32)
        try:
            outputs = self.model.run(None, {self.input_name: spectrograms_np})
            return torch.from_numpy(outputs[0])
        except Exception as e:
            print(f"❌ ONNX inference failed: {e}")
            print(f"Input shape: {spectrograms_np.shape}, expected: {self.input_info.shape}")
            raise

    # ---------------------- POSTPROCESS ----------------------

    def postprocess(self, output: torch.Tensor) -> List[Dict[str, float]]:
        results = []
        for pred in output:
            results.append({
                "ovrl_score": float(pred[0]),
                "sig_score": float(pred[1]),
                "bak_score": float(pred[2]),
                "mos": float(pred[0])
            })
        return results

    @property
    def sample_rate(self) -> int:
        return 16000
