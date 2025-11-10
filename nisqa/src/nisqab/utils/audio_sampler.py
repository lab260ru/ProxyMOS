import numpy as np
from typing import Dict, List, Iterator
from collections import defaultdict
from torch.utils.data import Sampler


class LengthBasedBatchSampler(Sampler):
    def __init__(self, 
                 file_paths: List[str], 
                 audio_lengths: Dict[str, float], 
                 batch_size: int = 32,
                 drop_last: bool = False,
                 shuffle: bool = True):

        super().__init__(None)
        
        self.file_paths = file_paths
        self.audio_lengths = audio_lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        self.file_to_idx = {file_path: idx for idx, file_path in enumerate(file_paths)}

        self.length_bins = self._create_length_bins()
        self.batches = self._create_batches()
        
    def _create_length_bins(self) -> Dict[str, List[int]]:
        bins = {f'{i}-{i+1}s': (i, i+1) for i in range(30)}
        bins['30s+'] = (30, float('inf'))
        
        grouped = defaultdict(list)
        
        for file_path in self.file_paths:
            duration = self.audio_lengths.get(file_path, 0.0)
            idx = self.file_to_idx[file_path]
    
            for bin_name, (min_dur, max_dur) in bins.items():
                if min_dur <= duration < max_dur:
                    grouped[bin_name].append(idx)
                    break
        
        return dict(grouped)
    
    def _create_batches(self) -> List[List[int]]:
        all_batches = []
        
        for bin_name, indices in self.length_bins.items():
            if len(indices) == 0:
                continue
            
            if self.shuffle:
                np.random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
 
        if self.shuffle:
            np.random.shuffle(all_batches)
            
        return all_batches
    
    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.batches)
    
    def __len__(self) -> int:
        return len(self.batches)
    
