import torch
from torch.utils.data import Dataset
import torchaudio
from typing import Optional, List, Dict, Callable, Any
from pathlib import Path
import os
import pandas as pd 


class AudioDataset(Dataset):
    """
    Dataset for audio files with resampling and optional transforms.
    
    Supports two input modes:
    - manifest: List[Dict] with 'audio_path' (legacy behavior)
    - audio_dir: Path to a directory containing audio files (auto-build manifest)
    """
    
    def __init__(
        self,
        manifest: Optional[List[Dict]] = None,
        audio_dir: Optional[str] = None,
        sample_rate: int = 16000,
        file_format: str = '.wav',
        time_length: Optional[float]= None,
        audio_transform: Optional[Callable] = None,
        text_transform: Optional[Callable] = None,
        text_key: str = 'transcript',
        recursive: bool = True,
        csv_file: Optional[str] = None
    ):
        """
        Initialize audio dataset.
        
        Args:
            manifest: List of dicts with 'audio_path' (optional when audio_dir is provided)
            audio_dir: Directory with audio files (optional when manifest is provided)
            sample_rate: Target sample rate for audio resampling
            file_format: Supported audio extension (e.g., '.wav')
            time_length: Maximum audio length in seconds (None to disable)
            audio_transform: Optional audio preprocessing function
            text_transform: Optional text preprocessing function (for future use)
            text_key: Key name for transcript in manifest
            recursive: Whether to search audio_dir recursively
        """
        supported_formats = ('.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac', '.wma')
        if file_format not in supported_formats:
            raise ValueError(f'File format {file_format} is not supported. Use one of: {supported_formats}')

        self.sample_rate = sample_rate
        self.time_length = time_length
        self.audio_transform = audio_transform
        self.text_transform = text_transform
        self.text_key = text_key
        
        self.file_format = file_format
        
            
        def _load_from_csv(self, csv_file: str) -> List[Dict]:
            """Load manifest from CSV file."""
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
            df = pd.read_csv(csv_file)
        
        # Проверяем наличие нужных колонок
            required_cols = ['audio_path', 'mos']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV missing required columns: {missing_cols}. Found: {df.columns.tolist()}")
            
            # Конвертируем в манифест
            manifest = []
            for _, row in df.iterrows():
                manifest.append({
                    'audio_path': str(row['audio_path']),
                    'mos': float(row['mos']),
                })
            
            return manifest
        
        if csv_file and not manifest:
            manifest = self._load_from_csv(csv_file)

        # Build manifest if audio_dir is provided
        if audio_dir and not manifest:
            base = Path(audio_dir)
            if not base.exists():
                raise FileNotFoundError(f"Audio directory not found: {base}")
            pattern = f"*{file_format}" if file_format.startswith('.') else f"*.{file_format}"
            files_iter = base.rglob(pattern) if recursive else base.glob(pattern)
            manifest = [{
                'audio_path': str(p),
                'transcript': ''
            } for p in files_iter if p.is_file()]

        if not manifest:
            raise ValueError("Manifest is empty or audio_dir not provided")
        
        
        if not manifest:
            raise ValueError("No files with MOS scores found")


        self.manifest = [
            item for item in manifest
            if item.get('audio_path')
            and item['audio_path'].lower().endswith(file_format)
            and os.path.exists(item['audio_path'])
        ]

        if len(self.manifest) == 0:
            raise FileNotFoundError("No valid audio files found for the dataset.")
        
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not isinstance(idx, int):
            raise AssertionError(f"Index must be integer, got {type(idx)}")
        sample = self.manifest[idx]
        sample_path = sample['audio_path']
        transcript = sample.get(self.text_key, None)
        mos = sample.get('mos', None)  # Получаем MOS если есть

        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Audio file not found: {sample_path}")
        
        waveform, original_sample_rate = torchaudio.load(sample_path)
        
        if original_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        if self.time_length:
            max_samples = int(self.time_length * self.sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            elif waveform.shape[1] < max_samples:
                pad_size = max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        
        if self.audio_transform:
            waveform = self.audio_transform(waveform)

        text_ids = None
        if self.text_transform and transcript:
            text_ids = self.text_transform(transcript)

        return {
            "waveform": waveform,            # torch.Tensor [C, T]
            "audio_path": sample_path,       # str - file path
            "transcript": transcript,        # optional transcript
            "text_ids": text_ids,           # optional tokenized text
            "mos": torch.tensor(mos, dtype=torch.float32) if mos is not None else None,  # MOS score
        }

