import torch
import hashlib
from typing import Optional


def _hash_tensor_to_uint64(tensor: torch.Tensor) -> int:
	"""
	Compute a stable 64-bit integer hash from a 1D float tensor.
	Hash is content-based and deterministic across runs and devices.
	"""
	# Ensure CPU, contiguous, float32 bytes for stable hashing
	buf = tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
	# Use a fast, stable hash
	digest = hashlib.blake2b(buf, digest_size=8).digest()
	# Interpret as unsigned 64-bit integer
	return int.from_bytes(digest, byteorder="big", signed=False)


def select_deterministic_window_batch(
	audio_batch: torch.Tensor,
	sample_rate: int,
	window_seconds: float = 10.0,
	deterministic: bool = True,
	seed: Optional[int] = None,
	pad_mode: str = "repeat",
) -> torch.Tensor:
	"""
	Select a window of fixed duration from each sample in a batch.

	Args:
		audio_batch: Tensor of shape [B, T] (mono)
		sample_rate: Sampling rate in Hz
		window_seconds: Desired window duration in seconds
		deterministic: If True, choose start by hashing content; else use PRNG
		seed: Optional seed for PRNG when deterministic is False
		pad_mode: How to handle short audio: "repeat" or "pad_zeros"

	Returns:
		Tensor [B, T_window] containing selected windows
	"""
	assert audio_batch.ndim == 2, f"Expected [B, T], got {audio_batch.shape}"
	B, T = audio_batch.shape
	window_len = int(round(window_seconds * sample_rate))
	if window_len <= 0:
		raise ValueError("window_seconds must be > 0")

	if T == window_len:
		return audio_batch

	# Handle short audio
	if T < window_len:
		if pad_mode == "repeat":
			# Tile then trim to exact length (avoids silence bias)
			repeats = (window_len + T - 1) // T
			out = audio_batch.repeat(1, repeats)[:, :window_len]
			return out
		elif pad_mode == "pad_zeros":
			pad = window_len - T
			return torch.nn.functional.pad(audio_batch, (0, pad))
		else:
			raise ValueError(f"Unknown pad_mode: {pad_mode}")

	# Long audio → choose start index
	max_start = T - window_len
	starts = []
	if deterministic:
		for i in range(B):
			s_h = _hash_tensor_to_uint64(audio_batch[i])
			starts.append(int(s_h % (max_start + 1)))
		starts = torch.tensor(starts, dtype=torch.long)
	else:
		gen = torch.Generator(device=audio_batch.device)
		if seed is not None:
			gen.manual_seed(seed)
		starts = torch.randint(0, max_start + 1, (B,), generator=gen, device=audio_batch.device)

	# Gather windows
	out = []
	for i in range(B):
		start = int(starts[i].item())
		out.append(audio_batch[i, start:start + window_len])
	return torch.stack(out, dim=0)


