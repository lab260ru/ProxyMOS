import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any


def compact_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching audio data with variable lengths.
    Pads waveforms and text_ids to the same length within the batch.

    Args:
        batch: List of dictionaries containing:
            - waveform: torch.Tensor [C, T] - audio waveform
            - audio_path: str - path to audio file
            - transcript: str - text transcript (optional)
            - text_ids: torch.Tensor - processed text tokens (optional)

    Returns:
        Dictionary containing batched data:
            - waveform: torch.Tensor [B, C, T] - batched audio waveforms
            - audio_path: List[str] - list of audio file paths
            - transcript: List[str|None] - list of text transcripts (or None)
            - text_ids: torch.Tensor [B, T] - batched text tokens (or None)
    """
    # Waveforms: convert all to [C, T] and pad
    waveforms = []
    for item in batch:
        waveform = item["waveform"]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
        waveforms.append(waveform.T)  # [T, C] for pad_sequence

    padded_waveforms = pad_sequence(waveforms, batch_first=True)  # [B, T, C]
    padded_waveforms = padded_waveforms.permute(0, 2, 1)           # [B, C, T]


    return {
        "waveform": padded_waveforms,                # [B, C, T]
        "audio_path": [item["audio_path"] for item in batch],
    }


