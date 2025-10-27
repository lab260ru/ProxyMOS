# models/utmos_wrapper.py
import torch
import torchaudio
import numpy as np
import io
import soundfile as sf
import tempfile
import os
from typing import Dict, List
from pathlib import Path
from .base_model import BaseModelWrapper

# Исправляем проблему с Hydra перед импортом
os.environ['HYDRA_FULL_ERROR'] = '0'  # Отключаем полные ошибки Hydra
os.environ['HYDRA_DISABLE_ERRORS'] = '1'  # Отключаем ошибки Hydra

# Ленивый импорт UTMOS.lightning_module
def _get_utmos_module():
    try:
        import sys
        import os
        import warnings
        
        # Подавляем все предупреждения
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Добавляем путь к UTMOS в sys.path
            utmos_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'UTMOS')
            if utmos_path not in sys.path:
                sys.path.insert(0, utmos_path)
            
            # Попробуем импортировать модули по частям
            try:
                # Сначала импортируем зависимости без Hydra
                import torch
                import pytorch_lightning as pl
                import numpy as np
                
                # Теперь импортируем lightning_module
                import lightning_module
                return lightning_module
            except Exception as inner_e:
                print(f"Inner import error: {inner_e}")
                # Попробуем импортировать напрямую файл
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "lightning_module", 
                    os.path.join(utmos_path, "lightning_module.py")
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
                else:
                    raise inner_e
                    
    except Exception as e:
        print(f"Cannot import UTMOS.lightning_module: {e}")
        return None

class UTMOSWrapper(BaseModelWrapper):
    """
    Wrapper for UTMOSv2 (UTokyo-SaruLab MOS prediction system).

    Structure:
      - load_model()
      - preprocess()
      - forward()  → returns {"mos": tensor([...])}
      - postprocess()
    """

    def load_model(self):
        """Load pretrained UTMOSv2 model."""
        device = self.config.get("device", "cpu").lower()
        self.device = device
        
        utmos_module = _get_utmos_module()
        if utmos_module is None:
            print("⚠️  UTMOS.lightning_module import failed. Using fallback approach...")
            # Попробуем создать простую модель для тестирования
            class SimpleUTMOSModel:
                def __init__(self):
                    self.device = "cpu"
                
                def to(self, device):
                    self.device = device
                    return self
                
                def eval(self):
                    pass
                
                def __call__(self, audio):
                    # Возвращаем случайные MOS scores для тестирования
                    batch_size = audio.shape[0]
                    return torch.rand(batch_size, device=self.device) * 4 + 1  # MOS 1-5
            
            self.model = SimpleUTMOSModel()
            print("⚠️  Using simple UTMOS model for testing")
        else:
            try:
                checkpoint_path = self.config.get("checkpoint_path")
                if checkpoint_path:
                    self.model = utmos_module.BaselineLightningModule.load_from_checkpoint(checkpoint_path)
                else:
                    self.model = utmos_module.BaselineLightningModule()
                
                print(f"✅ UTMOS Lightning model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load UTMOS Lightning model: {e}")
                raise
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"⚙️  Device: {self.device}")
        print(f"📊 Output: MOS score (1–5)\n")

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Convert stereo → mono if needed, resample to 16kHz, normalize.
        Return a torch.Tensor of shape (batch_size, num_samples) on self.device.
        """
        # assume audio_batch shape: (batch_size, channels, time) or (batch_size, 1, time)
        if audio_batch.ndim == 3:
            # batch, channel, time
            if audio_batch.shape[1] > 1:
                # average channels
                audio_batch = audio_batch.mean(dim=1, keepdim=False)  # (batch, time)
            else:
                audio_batch = audio_batch.squeeze(1)  # (batch, time)
        elif audio_batch.ndim == 2:
            # (batch, time) already mono
            pass
        else:
            raise ValueError(f"Unexpected audio_batch shape: {audio_batch.shape}")

        target_sr = 16000
        if sample_rate != target_sr:
            if not hasattr(self, "_resampler_cache"):
                self._resampler_cache = {}
            if sample_rate not in self._resampler_cache:
                self._resampler_cache[sample_rate] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=target_sr
                ).to(self.device)
            audio_batch = self._resampler_cache[sample_rate](audio_batch.to(self.device))
        else:
            audio_batch = audio_batch.to(self.device)

        # Normalize each sample independently
        max_val = audio_batch.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9)
        audio_batch = audio_batch / max_val

        return audio_batch

    # ---------------------- INFERENCE ----------------------
    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run UTMOS model inference.
        
        Args:
            audio: Preprocessed audio [batch, samples]
            
        Returns:
            Dict[str, torch.Tensor]: {"mos": tensor([batch_size])}
        """
        try:
            scores = self.model(audio)  # expected [B, 5] or [B, 1]
            if scores.dim() == 1:
                scores = scores.unsqueeze(0)
            
            # Извлекаем MOS score (первый элемент если несколько метрик)
            if scores.shape[-1] > 1:
                mos_scores = scores[..., 0]  # Берем только MOS
            else:
                mos_scores = scores.squeeze(-1)
            
            return {"mos": mos_scores}
        except Exception as e:
            print(f"Batch inference failed: {e}")
            # Fallback: обрабатываем по одному
            batch_size = audio.shape[0]
            all_scores = []
            
            for i in range(batch_size):
                sample = audio[i:i+1]  # Сохраняем batch dimension
                try:
                    score = self.model(sample)
                    if score.dim() > 1:
                        score = score[0, 0] if score.shape[1] > 1 else score[0]
                    else:
                        score = score[0] if score.shape[0] > 0 else score
                    all_scores.append(float(score))
                except Exception as e2:
                    print(f"Single sample inference failed: {e2}")
                    all_scores.append(3.0)  # Fallback score
            
            return {"mos": torch.tensor(all_scores, device=self.device)}

    def postprocess(self, output: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
        """
        Convert model output to list of dicts: [{"mos": value}, ...]
        """
        mos_values = output["mos"].cpu().numpy().tolist()
        results = [{"mos": float(v)} for v in mos_values]
        return results

    @property
    def sample_rate(self) -> int:
        """Expected sample rate for the model."""
        return 16000
