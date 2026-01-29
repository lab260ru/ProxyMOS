import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd


class AudioDataset(Dataset):


    def __init__(
        self,
        csv_file: Optional[str] = None,
        manifest: Optional[List[Dict]] = None,
        audio_dir: Optional[str] = None,
        sample_rate: int = 16000,
        time_length: Optional[float] = None,
        segment_seconds: Optional[float] = None,
        audio_transform: Optional[Callable] = None,
        recursive: bool = True,
    ):
        self.sample_rate = sample_rate
        self.audio_transform = audio_transform


        if csv_file:
            self.manifest = self._load_from_csv(csv_file)
        elif manifest:
            self.manifest = manifest
        elif audio_dir:
            self.manifest = self._build_from_dir(audio_dir, recursive)
        else:
            raise ValueError("csv_file, manifest or audio_dir must be provided")

        if len(self.manifest) == 0:
            raise ValueError("Dataset is empty")

        # -------- cached resamplers --------
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}


        self.segment_samples = (
            int(segment_seconds * sample_rate)
            if segment_seconds is not None
            else None
        )

        self.max_samples = (
            int(time_length * sample_rate)
            if time_length is not None
            else None
        )

    # ============================================================
    # Manifest loaders
    # ============================================================

    def _load_from_csv(self, csv_file: str) -> List[Dict]:
        if not os.path.exists(csv_file):
            raise FileNotFoundError(csv_file)

        df = pd.read_csv(csv_file)

        if not {"audio_path", "mos"}.issubset(df.columns):
            raise ValueError("CSV must contain columns: audio_path, mos")

        return [
            {
                "audio_path": str(row.audio_path),
                "mos": float(row.mos),
            }
            for row in df.itertuples(index=False)
        ]

    def _build_from_dir(self, audio_dir: str, recursive: bool) -> List[Dict]:
        base = Path(audio_dir)
        if not base.exists():
            raise FileNotFoundError(audio_dir)

        exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        files = base.rglob("*") if recursive else base.glob("*")

        return [
            {"audio_path": str(p), "mos": 0.0}
            for p in files
            if p.suffix.lower() in exts
        ]

    # ============================================================
    # Audio processing
    # ============================================================

    def _resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr == self.sample_rate:
            return waveform

        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate
            )

        return self._resamplers[orig_sr](waveform)

    def _process_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Segment-level crop + optional fixed-length pad/cut
        """
        T = waveform.shape[1]

        # ---- segment-level training ----
        if self.segment_samples is not None and T > self.segment_samples:
            start = torch.randint(
                0, T - self.segment_samples + 1, (1,)
            ).item()
            waveform = waveform[:, start:start + self.segment_samples]
            T = waveform.shape[1]

        # ---- fixed length ----
        if self.max_samples is not None:
            if T > self.max_samples:
                waveform = waveform[:, :self.max_samples]
            elif T < self.max_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.max_samples - T)
                )

        return waveform



    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.manifest[idx]
        path = sample["audio_path"]

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        waveform, sr = torchaudio.load(path)  # [C, T]

        waveform = self._resample(waveform, sr)
        waveform = self._process_length(waveform)

        if self.audio_transform is not None:
            waveform = self.audio_transform(waveform)

        return {
            "waveform": waveform,                       # [C, T]
            "mos": torch.tensor(sample["mos"], dtype=torch.float32),
            "audio_path": path,
        }

