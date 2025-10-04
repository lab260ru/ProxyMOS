import torch 
from  torch.utils.data import  Dataset
import  torchaudio
from typing import Optional, List, Dict, Callable, Any
import  os


class AudioDataset(Dataset):
    """
    Dataset for audio files with resampling and optional transforms.
    
    Returns:
        Dict containing waveform tensor, audio path, transcript, and text_ids.
    """
    def __init__(
        self,
        manifest: List[Dict],
        sample_rate:int,
        file_format:str = '.wav',
        time_lenght:Optional[float] = 4,
        audio_transform: Optional[Callable] = None,
        text_transform: Optional[Callable] = None,
        text_key: str = 'transcript',
    ):
        """
        Initialize audio dataset.
        
        Args:
            manifest: List of dicts with 'audio_path' key
            sample_rate: Target sample rate for audio
            file_format: Supported audio format (default: .wav)
            time_lenght: Max audio length in seconds (default: 4)
            audio_transform: Optional audio preprocessing function
            text_transform: Optional text preprocessing function
            text_key: Key name for text data in manifest (default: 'transcript')
        """
        if not isinstance(manifest, list) or not all("audio_path" in m for m in manifest):
            raise ValueError("Manifest need to  have key 'audio_path'")
        if file_format not in ('.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac', '.wma'):
            raise ValueError(f'File format:{file_format} is not supported')
        self.timestamp = time_lenght
        self.rate = sample_rate
        self.manifest = manifest
        self.audio_transform = audio_transform
        self.text_transform = text_transform
        self.text_key = text_key
        
        
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.manifest)
    
    
    def __getitem__(self, idx:int) ->Dict[str, Any]:
        """
        Get audio sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict with waveform, audio_path, transcript, and text_ids
        """
        assert(isinstance(idx, int))
        sample = self.manifest[idx]
        sample_path = sample['audio_path']
        transcript = sample.get(self.text_key, None) # use flexible text key  
        
        
        ### LOAD AND RESAMPLE
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Файл {sample_path} не найден")
        waveform, sample_rate = torchaudio.load(sample_path)
        
        if sample_rate != self.rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.rate
            )
            waveform = resampler(waveform)
            sample_rate = self.rate
            
            
        ### APPLY  TIMESTAMP
        if self.timestamp:
            max_samples = int(self.timestamp * self.rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            elif waveform.shape[1] < max_samples:
                pad_size = max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))
                
                
        ### TRANSFORM 
        if self.audio_transform:
            waveform = self.audio_transform(waveform)
        text_ids = None
        if self.text_transform and transcript:
            text_ids = self.text_transform(transcript)

        return {
            "waveform": waveform,            # torch.Tensor [C, T]
            "audio_path": sample_path,       # Filepath
            "transcript": transcript,        
            "text_ids": text_ids             
        }

