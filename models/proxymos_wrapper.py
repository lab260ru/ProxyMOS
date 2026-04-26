import torch
import torchaudio
from pathlib import Path
from typing import Dict, List
from proxymostrainer import OmniMOS

from .base_model import BaseModelWrapper
from accelerate import Accelerator

import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
set_seed(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


class ProxyMosWrapper(BaseModelWrapper):
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

    @staticmethod
    def load_checkpoint(checkpoint_path: str):
        import omnilingual_asr
        from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub

        hub = get_wav2vec2_model_hub()
        fs2_config = hub.get_model_config('omniASR_W2V_300M')
        model = hub.load_custom_model(
            Path(checkpoint_path),
            config=fs2_config,
            device=torch.device("cpu"),
        )
        return model, fs2_config

    def load_model(self):
        encoder, config_omni = self.load_checkpoint("omniASR-W2V-300M.pt")

        self.model = OmniMOS(encoder=encoder)
        state = torch.load(
            "/home/maxim/MOS_research/checkpoints/omniASR/best_model_full.pt",
            map_location="cpu"
        )
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        audio_batch = audio_batch.to(self.device, non_blocking=True)
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