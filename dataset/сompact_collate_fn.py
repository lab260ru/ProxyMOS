import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any


def compact_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching audio data with variable lengths.
    
    This function efficiently batches audio waveforms and text sequences by padding
    them to the same length within each batch.
    
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
    # Extract waveforms and ensure correct shape [T, C] for padding
    waveforms = []
    for item in batch:
        waveform = item["waveform"]
        # Ensure waveform is [C, T] format
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        # Transpose to [T, C] for padding
        waveforms.append(waveform.squeeze(0).T)
    
    # Handle text_ids with proper None handling
    text_ids = []
    for item in batch:
        if item.get("text_ids") is not None:
            text_ids.append(item["text_ids"])
        else:
            text_ids.append(torch.tensor([], dtype=torch.long))
    
    # Pad waveforms and transpose back to [B, C, T]
    padded_waveforms = pad_sequence(waveforms, batch_first=True)
    if padded_waveforms.dim() == 3:  # [B, T, C]
        padded_waveforms = padded_waveforms.permute(0, 2, 1)  # [B, C, T]
    elif padded_waveforms.dim() == 2:  # [B, T] - single channel
        padded_waveforms = padded_waveforms.unsqueeze(1)  # [B, 1, T]
    
    # Pad text_ids if any exist
    padded_text_ids = None
    if any(t.numel() > 0 for t in text_ids):
        padded_text_ids = pad_sequence(text_ids, batch_first=True, padding_value=0)
    
    return {
        "waveform": padded_waveforms,  # [B, C, T]
        "audio_path": [item["audio_path"] for item in batch],
        "transcript": [item.get("transcript") for item in batch],
        "text_ids": padded_text_ids
    }

