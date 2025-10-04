import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any

def compact_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    waveforms = [item["waveform"].squeeze(0).T for item in batch]  # [T, C]
    text_ids = [item["text_ids"] if item["text_ids"] is not None else torch.tensor([], dtype=torch.long) for item in batch]

    return {
        "waveform": pad_sequence(waveforms, batch_first=True).permute(0, 2, 1),  # [B, C, T]
        "audio_path": [item["audio_path"] for item in batch],
        "transcript": [item["transcript"] for item in batch],
        "text_ids": pad_sequence(text_ids, batch_first=True, padding_value=0) if any(t.numel() > 0 for t in text_ids) else None
    }

