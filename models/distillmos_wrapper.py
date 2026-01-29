import torch
import torchaudio
from pathlib import Path
from typing import Dict, List

from .base_model import BaseModelWrapper
import distillmos
from accelerate import Accelerator


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


class DistillmosWrapper(BaseModelWrapper):
    def __init__(self, config, device=None):
        super().__init__(config)

        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16")
        )
        self.device = self.accelerator.device

        self.resampler = torchaudio.transforms.Resample(
            new_freq=16000
        )
        self.use_compile = self.config.get("compile", False)

    def load_model(self):
        if self.model is not None:
            return

        model = distillmos.ConvTransformerSQAModel()
        model = model.to(self.device).eval()
        if  self.use_compile:
            self.model = torch.compile(
            self.model,
            mode="max-autotune",
            fullgraph=False
        )

        self.model = model

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        
        audio_batch.to(self.device, non_blocking=True)
        # audio_batch: [B, C, T]
        if audio_batch.shape[1] > 1:
            audio_batch = audio_batch.mean(dim=1)
        else:
            audio_batch = audio_batch[:, 0]

        if sample_rate != 16000:
            audio_batch = self.resampler(audio_batch)

        max_val = audio_batch.abs().amax(dim=1, keepdim=True)
        audio_batch = audio_batch / (max_val + 1e-9)

        return audio_batch

    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio = audio.squeeze(1)

        with self.accelerator.autocast():
            output = self.model(audio)

        return output

    def postprocess(self, output: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
        return [{"mos": float(v)} for v in output]

    @property
    def sample_rate(self) -> int:
        return 16000
