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

        model_path = self.config.get("primary_model_path") or self.config.get("checkpoint_path")
        if not model_path:
            raise ValueError("Either 'primary_model_path' or 'checkpoint_path' must be provided in config")

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

        print(f"✅ DNSMOS model loaded: {model_path.name}")
        print(f"Expected input shape: {self.input_info.shape}")
        print("This model expects spectrograms, not raw waveforms!\n")

    # ---------------------- PREPROCESS ----------------------

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Convert raw waveforms into spectrograms expected by DNSMOS.

        Args:
            audio_batch (torch.Tensor): Input tensor [batch, channels, samples]
            sample_rate (int): Original audio sample rate

        Returns:
            torch.Tensor: Spectrograms [batch, time_frames, freq_bins]
        """
        # Convert to mono
        if audio_batch.shape[1] > 1:
            audio_batch = audio_batch.mean(dim=1, keepdim=True)

        # Resample to 16 kHz if needed
        target_sr = 16000
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            audio_batch = resampler(audio_batch)

        # Normalize to [-1, 1]
        max_val = audio_batch.abs().max()
        audio_batch = audio_batch / (max_val + 1e-9)

        # Generate spectrograms
        spectrograms = []
        for i in range(audio_batch.shape[0]):
            audio = audio_batch[i, 0]
            spectrogram = self._audio_to_spectrogram(audio, target_sr)
            spectrograms.append(spectrogram)

        spectrograms = torch.stack(spectrograms)
        return spectrograms.to(self.device)

    # ---------------------- AUDIO -> SPECTROGRAM ----------------------

    def _audio_to_spectrogram(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Convert waveform into log-mel spectrogram.

        Args:
            audio (torch.Tensor): [samples]
            sample_rate (int): Sample rate

        Returns:
            torch.Tensor: [time_frames, freq_bins]
        """
        n_fft = 320
        hop_length = 160
        n_mels = 161  # slightly below n_freqs (161) to avoid all-zero filters

        # Ensure processing happens on the correct device
        audio = audio.to(self.device)

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        ).to(self.device)

        spec = mel_spec(audio)
        spec_db = torchaudio.transforms.AmplitudeToDB().to(self.device)(spec)
        spec_db = spec_db.T  # [time, freq]

        # Adjust to fixed frame length (901)
        target_time_frames = 901
        current_frames = spec_db.shape[0]
        if current_frames > target_time_frames:
            spec_db = spec_db[:target_time_frames, :]
        elif current_frames < target_time_frames:
            pad = target_time_frames - current_frames
            spec_db = torch.nn.functional.pad(spec_db, (0, 0, 0, pad), mode="constant", value=-80.0)

        return spec_db

    # ---------------------- INFERENCE ----------------------

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Run ONNX inference on spectrograms.

        Args:
            spectrograms (torch.Tensor): [batch, time_frames, freq_bins]

        Returns:
            torch.Tensor: [batch, 3] (ovrl, sig, bak)
        """
        print(f"Running inference on batch: {spectrograms.shape}")

        # Convert to numpy for ONNX runtime
        spectrograms_np = spectrograms.cpu().numpy().astype(np.float32)

        try:
            outputs = self.model.run(None, {self.input_name: spectrograms_np})
            return torch.from_numpy(outputs[0]).to(self.device)
        except Exception as e:
            print(f"❌ ONNX inference failed: {e}")
            print(f"Input shape: {spectrograms_np.shape}, expected: {self.input_info.shape}")
            raise

    # ---------------------- POSTPROCESS ----------------------

    def postprocess(self, output: torch.Tensor) -> List[Dict[str, float]]:
        """
        Convert model output to MOS metrics.

        Args:
            output (torch.Tensor): Model predictions [batch, 3]

        Returns:
            List[Dict[str, float]]: DNSMOS metrics
        """
        results = []
        for pred in output:
            results.append({
                "ovrl_score": float(pred[0]),  # Overall quality
                "sig_score": float(pred[1]),   # Speech signal quality
                "bak_score": float(pred[2]),   # Background noise quality
                "mos": float(pred[0])          # MOS alias for ovrl_score
            })
        return results

    # ---------------------- PROPERTY ----------------------

    @property
    def sample_rate(self) -> int:
        """DNSMOS always expects 16 kHz audio."""
        return 16000
