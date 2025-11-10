# models/nisqa_wrapper.py
import math
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from .base_model import BaseModelWrapper
from .audio_utils import select_deterministic_window_batch
from nisqa.src.nisqab.core.model_torch import model_init
from nisqa.src.nisqab.utils.audio_cache import create_audio_length_cache
from nisqa.src.nisqab.utils.audio_sampler import LengthBasedBatchSampler
import yaml

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

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

        checkpoint_path = self.config.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("'checkpoint_path' is required in config to load NISQA model")

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        # YAML config must be alongside checkpoint OR explicitly provided
        config_path = self.config.get("config_path")
        if config_path is None:
            config_path = checkpoint_path.with_suffix(".yaml")  # same name as checkpoint

        config_path = Path(config_path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML config exactly as is
        with open(config_path, "r") as f:
            model_cfg = yaml.safe_load(f)

        # Override checkpoint inside config
        model_cfg["ckp"] = str(checkpoint_path)
        model_cfg["inf_device"] = self.device

        # Initialize model using full config
        self.model = model_init(model_cfg).to(self.device, dtype=torch.bfloat16).eval()
        self.model_cfg = model_cfg

        mel_sr = int(model_cfg.get("ms_sr") or self.sample_rate)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=mel_sr,
            n_fft=int(model_cfg["ms_n_fft"]),
            hop_length=int(model_cfg["ms_hop_length"]),
            win_length=int(model_cfg["ms_win_length"]),
            window_fn=torch.hann_window,
            power=1.0,
            n_mels=int(model_cfg["ms_n_mels"]),
            f_min=0.0,
            f_max=float(model_cfg["ms_fmax"]),
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        ).to(self.device)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="magnitude",
            top_db=80.0,
        ).to(self.device)

        self.seg_length = int(model_cfg["ms_seg_length"])
        self.seg_hop = int(model_cfg["ms_seg_hop_length"])
        self.max_segments = model_cfg.get("ms_max_length")



  # ---------------------- PREPROCESS ----------------------

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        # Optional deterministic 10s windowing (default True)
        segment_seconds = float(self.config.get("segment_seconds", 10.0))
        deterministic = bool(self.config.get("deterministic_segment", True))
        segment_seed = self.config.get("segment_seed")
        # Select window before normalization to keep scale consistent across segments
        audio_batch = select_deterministic_window_batch(
            audio_batch=audio_batch,
            sample_rate=target_sr,
            window_seconds=segment_seconds,
            deterministic=deterministic,
            seed=segment_seed,
            pad_mode="repeat",
        )

        # Normalize to [-1, 1]
        max_val = audio_batch.abs().max(dim=-1, keepdim=True)[0]
        audio_batch = audio_batch / (max_val + 1e-9)
        
        features = []
        n_wins_list = []

        for audio in audio_batch:
            spec = self.mel_transform(audio.unsqueeze(0)).squeeze(0)
            spec = self.db_transform(spec)

            segs, n_wins = self._segment_spectrogram(spec)
            features.append(segs)
            n_wins_list.append(n_wins)

        max_segments = max(feat.shape[0] for feat in features)
        padded = []
        for feat in features:
            if feat.shape[0] < max_segments:
                pad_tensor = torch.zeros(
                    (max_segments, feat.shape[1], feat.shape[2], feat.shape[3]),
                    device=self.device,
                    dtype=feat.dtype,
                )
                pad_tensor[:feat.shape[0]] = feat
                feat = pad_tensor
            padded.append(feat)

        x_batch = torch.stack(padded, dim=0)
        n_wins_tensor = torch.tensor(n_wins_list, device=self.device, dtype=torch.long)

        return x_batch, n_wins_tensor

    def _segment_spectrogram(self, spec: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if self.seg_length % 2 == 0:
            raise ValueError(f"seg_length must be odd (got {self.seg_length})")

        if spec.dim() != 2:
            raise ValueError(f"Expected spectrogram with shape [mel, time], got {spec.shape}")

        mel_bins, num_frames = spec.shape
        if num_frames < self.seg_length:
            pad_frames = self.seg_length - num_frames
            spec = F.pad(spec, (0, pad_frames))
            num_frames = spec.shape[1]

        n_wins = num_frames - (self.seg_length - 1)
        if n_wins < 1:
            n_wins = 1

        device = spec.device
        idx1 = torch.arange(self.seg_length, device=device)
        idx2 = torch.arange(n_wins, device=device)
        idx = idx2.unsqueeze(1) + idx1.unsqueeze(0)
        windows = spec.transpose(1, 0)[idx, :].unsqueeze(1).transpose(3, 2)

        if self.seg_hop > 1:
            windows = windows[:: self.seg_hop]
            n_wins = math.ceil(n_wins / self.seg_hop)

        if self.max_segments:
            max_segments = int(self.max_segments)
            if windows.shape[0] > max_segments:
                windows = windows[:max_segments]
                n_wins = min(n_wins, max_segments)
            elif windows.shape[0] < max_segments:
                pad_windows = torch.zeros(
                    (max_segments, windows.shape[1], windows.shape[2], windows.shape[3]),
                    device=device,
                    dtype=windows.dtype,
                )
                pad_windows[: windows.shape[0]] = windows
                windows = pad_windows

        n_wins = min(int(n_wins), windows.shape[0])

        return windows.to(self.device), n_wins

    # ---------------------- INFERENCE ----------------------
    @torch.inference_mode()
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Run NISQA model inference.
        
        Args:
            audio: Preprocessed audio [batch, samples]
            
        Returns:
            torch.Tensor: [batch, 5] predictions
                          (MOS, Noisiness, Discontinuity, Coloration, Loudness)
        """
        x_batch, n_wins = inputs
        x_batch = x_batch.to(self.device)
        n_wins = n_wins.to(self.device)

        try:
            scores = self.model(x_batch, n_wins)
            if scores.dim() == 1:
                scores = scores.unsqueeze(0)
            return scores
        except Exception:
            batch_size = x_batch.shape[0]
            all_scores = []
            for i in range(batch_size):
                sample_x = x_batch[i : i + 1]
                sample_n = n_wins[i : i + 1]
                s = self.model(sample_x, sample_n)
                if s.dim() == 1:
                    s = s.unsqueeze(0)
                all_scores.append(s)
            return torch.cat(all_scores, dim=0)

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




