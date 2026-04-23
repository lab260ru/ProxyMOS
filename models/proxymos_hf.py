import torch
import torchaudio
from pathlib import Path
from typing import Dict, List
from proxymostrainer import OmniMOS

from .base_model import BaseModelWrapper
from accelerate import Accelerator
from transformers import Wav2Vec2Model
from encoders import build_encoder

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


class ProxyMosWrapper_HF(BaseModelWrapper):
    def __init__(self, config, device=None):
        super().__init__(config)

        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16")
        )
        self.device = self.accelerator.device

        self.resampler = torchaudio.transforms.Resample(
            new_freq=16000
        )
        print(config)
    @staticmethod
    def load_checkpoint_hf(checkpoint_path: str):   
        hf_model =   Wav2Vec2Model.from_pretrained(checkpoint_path, trust_remote_code=True , token = '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')
        hf_model.eval()
        return  hf_model
            
        


    def load_model(self):
         print('loading  model')
         encoder = self.load_checkpoint_hf('ylacombe/omniASR_W2V_300M_SSL')
         encoder = build_encoder("huggingface", encoder)
         self.model = OmniMOS(encoder=encoder)
         state_model = torch.load('checkpoints/omniASR/hf/best_model_half.pt', map_location="cpu", weights_only=True)
         self.model.load_state_dict(state_model)
         self.model = self.model.to(self.device)
         return  self.model.eval()
        


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