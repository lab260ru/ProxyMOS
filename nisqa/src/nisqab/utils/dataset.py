from ast import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict
import torchaudio

class NISQADataset(Dataset):    
    def __init__(
        self,
        file_paths,
        audio_lengths: Dict[str, float] = None,
        seg_length=15,
        seg_hop=1,
        max_length=2000,
        n_fft=4096,
        hop_length=0.01,
        win_length=0.02,
        n_mels=48,
        fmax=20000
    ):

        self.file_paths = file_paths
        self.audio_lengths = audio_lengths or {}
        self.seg_length = seg_length
        self.seg_hop = seg_hop
        self.max_length = max_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmax = fmax
        self.target_sr = 48000

        self.hop_length = int(self.target_sr * self.hop_length)
        self.win_length = int(self.target_sr * self.win_length)
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window, 
            center=True,
            pad_mode="reflect",
            power=1.0, 
            n_mels=self.n_mels,
            f_min=0.0,
            f_max=self.fmax,
            norm='slaney',        
            mel_scale='slaney',     # (или htk=False)
            onesided=True 
        )

    def get_torchaudio_melspec(self, file_path: str) -> torch.Tensor:
        '''Calculate mel-spectrograms with torchaudio'''
        try:
            wav, sr = torchaudio.load(file_path)
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.target_sr)
        except:
            raise ValueError('Could not load file {}'.format(file_path))
        
        if wav.ndim == 2:
            if wav.shape[0] == 1:
                wav = wav.squeeze(0)
            else:
                wav = wav.mean(dim=0)
                
        # Convert to specified sample rate
        if wav.shape[0] < self.target_sr * 0.1:  # Very short file
            wav = torch.nn.functional.pad(wav, (0, int(self.target_sr * 0.1) - wav.shape[0]))

        S = self.mel_transform(wav.unsqueeze(0)).squeeze(0)
        spec = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)(S)

        return spec
    
    def segment_specs(self, x: torch.Tensor) -> Tuple(torch.Tensor, torch.Tensor):
        '''Segment a spectrogram into "seg_length" wide spectrogram segments'''
        if self.seg_length % 2 == 0:
            raise ValueError('seg_length must be odd! (seg_lenth={})'.format(self.seg_length))
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        n_wins = x.shape[1] - (self.seg_length - 1)
        if n_wins < 1:
            raise ValueError(
                f"Sample too short. Only {x.shape[1]} windows available but seg_length={self.seg_length}. "
                f"Consider zero padding the audio sample."
            )

        # broadcast magic to segment melspec
        idx1 = torch.arange(self.seg_length)
        idx2 = torch.arange(n_wins)
        idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
        x = x.transpose(1,0)[idx3,:].unsqueeze(1).transpose(3,2)
            
        if self.seg_hop > 1:
            x = x[::self.seg_hop,:]
            n_wins = int(np.ceil(n_wins / self.seg_hop))
            
        if self.max_length is not None:
            if self.max_length < n_wins:
                raise ValueError(f'n_wins {n_wins} > max_length {self.max_length}. Increase max window length ms_max_segments!')
            x_padded = torch.zeros((self.max_length, x.shape[1], x.shape[2], x.shape[3]))
            x_padded[:n_wins,:] = x
            x = x_padded
                    
        return x, n_wins
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        spec = self.get_torchaudio_melspec(file_path)
        x_seg, n_wins = self.segment_specs(spec)
        
        return {
            'file_path': file_path,
            'x_spec_seg': x_seg,
            'n_wins': n_wins if isinstance(n_wins, torch.Tensor) else torch.tensor(n_wins)
        }


def collate_fn(batch):
    file_paths = [item['file_path'] for item in batch]
    x_specs = [item['x_spec_seg'] for item in batch]
    n_wins = [item['n_wins'] for item in batch]
    
    max_segments = max([x.shape[0] for x in x_specs])

    batch_x = []
    batch_n_wins = []
    
    for x, n in zip(x_specs, n_wins):
        if x.shape[0] < max_segments:
            x_padded = torch.zeros((max_segments, x.shape[1], x.shape[2], x.shape[3]))
            x_padded[:x.shape[0], :, :, :] = x
            x = x_padded
        
        # batch dimension: [num_segments, 1, mel_bins, seg_length] -> [1, num_segments, 1, mel_bins, seg_length]
        batch_x.append(x.unsqueeze(0))
        batch_n_wins.append(n)
    
    #  [batch_size, max_segments, 1, mel_bins, seg_length]
    x_batch = torch.cat(batch_x, dim=0)
    n_wins_batch = torch.stack(batch_n_wins)
    
    return {
        'file_paths': file_paths,
        'x_spec_seg': x_batch,
        'n_wins': n_wins_batch
    }
