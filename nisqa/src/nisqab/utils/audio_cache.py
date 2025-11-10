import os
import json
from typing import Dict, List
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch 
import torchaudio


def measure_audio_length(file_path: str) -> tuple:
    try:
        info = torchaudio.info(file_path)
        duration = info.num_frames / info.sample_rate
        return file_path, duration
    except Exception as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return file_path, 0.0


def create_audio_length_cache(
    file_paths: List[str],
    cache_file: str = None,
    num_workers: int = None,
    force_rebuild: bool = False
    ) -> Dict[str, float]:

    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    cache = {}
    if cache_file and os.path.exists(cache_file) and not force_rebuild:
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            print(f"Loaded existing cache with {len(cache)} entries from {cache_file}")
        except Exception as e:
            print(f"Could not load cache: {e}")
            cache = {}
    
    missing_files = [f for f in file_paths if f not in cache]
    
    if missing_files:
        print(f"Measuring audio lengths for {len(missing_files)} files using {num_workers} workers...")
        
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(measure_audio_length, missing_files), 
                total=len(missing_files),
                desc="Measuring audio lengths"
            ))
        

        for file_path, duration in results:
            cache[file_path] = duration
        
        if cache_file:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=2)
                print(f"Saved cache with {len(cache)} entries to {cache_file}")
            except Exception as e:
                print(f"Could not save cache: {e}")
    
    return {f: cache.get(f, 0.0) for f in file_paths}