import torch
from typing import Dict, List
from .base_model import BaseModelWrapper


class XLSRSQAWrapper(BaseModelWrapper):
    """
    Wrapper для XLS-R SQA без reference.
    Всегда переводит модель в bfloat16 при наличии CUDA.
    """

    def load_model(self):
        from xls_r_sqa.e2e_model import E2EModel
        from xls_r_sqa.config import XLSR_2B_TRANSFORMER_32DEEP_CONFIG

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем модель
        self.model = E2EModel(
            config=XLSR_2B_TRANSFORMER_32DEEP_CONFIG,
            xlsr_layers=10,
            auto_download=True,
        ).to(self.device)

        # Если есть CUDA — принудительно bf16
        self.use_bf16 = torch.cuda.is_available()
        if self.use_bf16:
            self.model = self.model.to(dtype=torch.bfloat16)

        self.model.eval()

        # XLS-R требует 16 kHz
        self._sample_rate = 16000

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Приводим к [B, T] + ресэмплинг + нормализация
        """
        if audio_batch.dim() == 3 and audio_batch.shape[1] > 1:
            audio_batch = audio_batch.mean(dim=1, keepdim=True)

        audio = audio_batch.squeeze(1).to(self.device)

        # Ресэмплинг
        if sample_rate != self._sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self._sample_rate
            ).to(self.device)
            audio = resampler(audio)

        # Peak-normalize per example
        max_val = audio.abs().amax(dim=-1, keepdim=True) + 1e-9
        audio = (audio / max_val).contiguous()

        return audio

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Предсказание MOS
        """
        audio = audio.to(self.device)

        # bf16 autocast если есть CUDA
        amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float32

        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype,
            enabled=self.use_bf16
        ), torch.no_grad():
            scores = self.model(audio)

        return scores

    def postprocess(self, output: torch.Tensor) -> List[Dict[str, float]]:
        """
        Возвращает список нормальных значений
        """
        return [
            {"mos_score": float(v)}
            for v in output.detach().cpu().flatten()
        ]

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
