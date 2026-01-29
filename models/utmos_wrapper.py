import torch
import omegaconf
import typing
import collections

# Патч для torch.load
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    torch.serialization.add_safe_globals([
        omegaconf.dictconfig.DictConfig,
        omegaconf.listconfig.ListConfig,
        omegaconf.base.ContainerMetadata,
        typing.Any,
        dict,
        collections.defaultdict,
        collections.OrderedDict,
        collections.Counter
    ])
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

import torch
import torchaudio
import numpy as np
import os
from typing import Dict, List
from .base_model import BaseModelWrapper
import sys
import pytorch_lightning as pl
from accelerate import Accelerator

# Добавляем путь к UTMOS
sys.path.append('/home/maxim/MOS_research/UTMOS')

# Импортируем после добавления пути
try:
    import UTMOS.lightning_module as lightning_module
    import UTMOS.model as utmos_model
except ImportError as e:
    print(f"Error importing UTMOS: {e}")
    print("Make sure UTMOS is in the correct path: /home/maxim/MOS_research/UTMOS")
    raise

# Оптимизации CUDA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

class UTMOSWrapper(BaseModelWrapper):
    """
    Optimized wrapper for UTMOSv2.
    Architecturally aligned with DistillMOSWrapper.
    """

    def __init__(self, config, device=None):
        super().__init__(config)

        # Получаем путь к SSL модели из конфига
        self.ssl_model_path = config.get(
            "ssl_model_path",
            "/home/maxim/MOS_research/UTMOS/wav2vec_small.pt"
        )
        
        # Проверяем существование файла
        if not os.path.exists(self.ssl_model_path):
            raise FileNotFoundError(
                f"SSL model checkpoint not found: {self.ssl_model_path}\n"
                f"Please ensure the checkpoint file exists and the path is correct."
            )
        
        print(f"✓ SSL model path: {self.ssl_model_path}")

        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16")
        )
        self.device = self.accelerator.device

        self.resamplers = {}
        self.model = None

    # ---------------------- MODEL ----------------------
    def load_model(self):
        if self.model is not None:
            return

        # Патчим функцию load_ssl_model перед загрузкой модели
        self._patch_ssl_loader()

        # Получаем путь к чекпоинту
        checkpoint_path = self.config.get("checkpoint_path", "epoch3D7459.ckpt")
        
        # Проверяем существование чекпоинта
        if not os.path.exists(checkpoint_path):
            # Ищем в текущей директории
            current_dir = os.getcwd()
            checkpoint_in_current = os.path.join(current_dir, checkpoint_path)
            if os.path.exists(checkpoint_in_current):
                checkpoint_path = checkpoint_in_current
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}\n"
                    f"Also tried: {checkpoint_in_current}"
                )
        
        print(f"✓ Loading checkpoint from: {checkpoint_path}")

        # Создаем конфиг для UTMOS
        cfg = {
            "ssl_model_path": self.ssl_model_path,
            "segment_seed": self.config.get("segment_seed", 12345)
        }

        # Загружаем модель
        try:
            self.model = lightning_module.BaselineLightningModule.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                cfg=cfg,
                map_location=self.device,
                strict=False  # Разрешаем пропускать отсутствующие ключи
            )
        except Exception as e:
            print(f"Error loading UTMOS model: {e}")
            print("Trying with a different approach...")
            # Альтернативный способ загрузки
            self._load_model_alternative(checkpoint_path, cfg)

        self.model.eval()
        self.model.to(self.device)

        print(f"⚙️  UTMOS loaded on {self.device}")
        print("📊 Output: MOS score (1–5)")

    def _patch_ssl_loader(self):
        """Патчим функцию load_ssl_model для использования правильного пути"""
        
        original_load_ssl_model = utmos_model.load_ssl_model
        
        def patched_load_ssl_model(cp_path=None):
            # Если путь не указан или файл не существует, используем наш путь
            if cp_path is None or not os.path.exists(cp_path):
                cp_path = self.ssl_model_path
            
            print(f"Loading SSL model from: {cp_path}")
            return original_load_ssl_model(cp_path)
        
        # Применяем патч
        utmos_model.load_ssl_model = patched_load_ssl_model
        
        # Также патчим в lightning_module, если нужно
        if hasattr(lightning_module, 'load_ssl_model'):
            lightning_module.load_ssl_model = patched_load_ssl_model

    def _load_model_alternative(self, checkpoint_path, cfg):
        """Альтернативный способ загрузки модели"""
        # Создаем экземпляр модели
        self.model = lightning_module.BaselineLightningModule(cfg=cfg)
        
        # Загружаем веса
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Адаптируем ключи, если нужно
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Удаляем префиксы, если они есть
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        # Загружаем state_dict
        self.model.load_state_dict(new_state_dict, strict=False)

    # ---------------------- PREPROCESS ----------------------
    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        audio_batch: [B, C, T] or [B, T]
        returns: [B, T] @ 16kHz
        """
        audio_batch = audio_batch.to(self.device, non_blocking=True)

        # Mono
        if audio_batch.dim() == 3:
            audio_batch = audio_batch.mean(dim=1)

        # Resample
        if sample_rate != 16000:
            if sample_rate not in self.resamplers:
                self.resamplers[sample_rate] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000
                ).to(self.device)

            audio_batch = self.resamplers[sample_rate](audio_batch)

        # Normalize
        max_val = audio_batch.abs().amax(dim=1, keepdim=True)
        audio_batch = audio_batch / (max_val + 1e-9)

        return audio_batch

    # ---------------------- INFERENCE ----------------------
    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        audio: [B, T]
        returns: {"mos": [B]}
        """
        B = audio.size(0)

        # UTMOS expects [B, 1, T]
        audio = audio.unsqueeze(1)

        batch = {
            "wav": audio,
            "domains": torch.zeros(B, device=self.device, dtype=torch.long),
            "judge_id": torch.full((B,), 288, device=self.device, dtype=torch.long),
        }

        with self.accelerator.autocast():
            output = self.model(batch)

        # output: [B, T] or [B, T, 1]
        if output.dim() == 3:
            output = output.squeeze(-1)

        mos = output.mean(dim=1)

        # Scale to MOS range
        mos = mos * 2.0 + 3.0

        return {"mos": mos.float()}

    # ---------------------- POSTPROCESS ----------------------
    def postprocess(self, output: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
        mos = output["mos"].cpu().numpy().tolist()
        return [{"mos": float(v)} for v in mos]

    # ---------------------- META ----------------------
    @property
    def sample_rate(self) -> int:
        return 16000
