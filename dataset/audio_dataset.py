import torch
from torch.utils.data import Dataset
import torchaudio
from typing import Optional, List, Dict, Callable, Any
import os


class AudioDataset(Dataset):
    """
    Dataset for audio files with resampling and optional transforms.
    
    This dataset loads audio files from a manifest, resamples them to a target sample rate,
    applies optional transforms, and returns standardized data structures.
    
    Returns:
        Dict containing waveform tensor, audio path, transcript, and text_ids.
    """
    
    def __init__(
        self,
        manifest: List[Dict],
        sample_rate: int,
        file_format: str = '.wav',
        time_length: Optional[float] = 4,
        audio_transform: Optional[Callable] = None,
        text_transform: Optional[Callable] = None,
        text_key: str = 'transcript',
    ):
        """
        Initialize audio dataset.
        
        Args:
            manifest: List of dictionaries with 'audio_path' key containing file paths
            sample_rate: Target sample rate for audio resampling
            file_format: Supported audio format (default: '.wav')
            time_length: Maximum audio length in seconds (default: 4)
            audio_transform: Optional audio preprocessing function
            text_transform: Optional text preprocessing function
            text_key: Key name for text data in manifest (default: 'transcript')
            
        Raises:
            ValueError: If manifest format is invalid or file format is unsupported
        """
        if not isinstance(manifest, list) or not all("audio_path" in m for m in manifest):
            raise ValueError("Manifest must be a list with 'audio_path' key in each item")
        
        supported_formats = ('.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac', '.wma')
        if file_format not in supported_formats:
            raise ValueError(f'File format {file_format} is not supported. Use one of: {supported_formats}')
        
        self.time_length = time_length
        self.sample_rate = sample_rate
        self.manifest = manifest
        self.audio_transform = audio_transform
        self.text_transform = text_transform
        self.text_key = text_key
        
        
    def __len__(self) -> int:
        """
        Return number of samples in dataset.
        
        Returns:
            Number of audio files in the manifest
        """
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get audio sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing:
                - waveform: torch.Tensor [C, T] - audio waveform
                - audio_path: str - path to audio file
                - transcript: str - text transcript (if available)
                - text_ids: torch.Tensor - processed text tokens (if available)
                
        Raises:
            FileNotFoundError: If audio file doesn't exist
            AssertionError: If idx is not an integer
        """
        if not isinstance(idx, int):
            raise AssertionError(f"Index must be integer, got {type(idx)}")
        
        sample = self.manifest[idx]
        sample_path = sample['audio_path']
        transcript = sample.get(self.text_key, None)
        
        # Load and resample audio
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Audio file not found: {sample_path}")
        
        waveform, original_sample_rate = torchaudio.load(sample_path)
        
        # Resample if necessary
        if original_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate, 
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Apply time length constraint
        if self.time_length:
            max_samples = int(self.time_length * self.sample_rate)
            if waveform.shape[1] > max_samples:
                # Truncate if too long
                waveform = waveform[:, :max_samples]
            elif waveform.shape[1] < max_samples:
                # Pad if too short
                pad_size = max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        
        # Apply transforms
        if self.audio_transform:
            waveform = self.audio_transform(waveform)
        
        text_ids = None
        if self.text_transform and transcript:
            text_ids = self.text_transform(transcript)

        return {
            "waveform": waveform,            # torch.Tensor [C, T]
            "audio_path": sample_path,       # str - file path
            "transcript": transcript,        # str - text transcript
            "text_ids": text_ids             # torch.Tensor - processed text
        }

