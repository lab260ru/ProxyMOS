# models/whisqa_wrapper.py
import torch
import torchaudio
from typing import Dict, List, Optional
from .base_model import BaseModelWrapper


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


class WhiSQAWrapper(BaseModelWrapper):
    """
    Wrapper for WhiSQA (Whisper-based Speech Quality Assessment)
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        self.resampler = torchaudio.transforms.Resample(new_freq=16000)
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16")
        )
        self.device = self.accelerator.device
        self.use_compile = self.config.get("compile", False)

    # ---------------------- LOAD MODEL ----------------------

    def load_model(self):
        from WhiSQA.models.whisper_ni_predictors import (
            whisperMetricPredictorEncoderLayersTransformerSmall,
            whisperMetricPredictorEncoderLayersTransformerSmalldim,
        )

        checkpoint_path = self.config.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("'checkpoint_path' is required")

        model_type = self.config.get("model_type", "multi")

        if model_type == "single":
            model = whisperMetricPredictorEncoderLayersTransformerSmall()
        elif model_type == "multi":
            model = whisperMetricPredictorEncoderLayersTransformerSmalldim()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        model.load_state_dict(state, strict=True)

        model.to(self.device, dtype=self.dtype)
        model.eval()

        if self.use_compile:
            self.model = torch.compile(
            self.model,
            mode="max-autotune",
            fullgraph=False
        )
        self.model = model



    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        audio_batch: [B, C, T]
        """

        audio_batch = audio_batch.to(
            self.device,
            non_blocking=True,
        )


        if audio_batch.shape[1] > 1:
            audio_batch = audio_batch.mean(dim=1, keepdim=True)


        if sample_rate != self.target_sr:
            audio_batch = self.resampler(audio_batch)


        max_val = audio_batch.abs().amax(dim=-1, keepdim=True)
        audio_batch = audio_batch / (max_val.clamp_min(1e-9))

        return audio_batch

    # ---------------------- INFERENCE ----------------------

    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio = audio.squeeze(1)

        with self.accelerator.autocast():
            output = self.model(audio)

        return output


    # ---------------------- POSTPROCESS ----------------------

    def postprocess(self, output: torch.Tensor) -> List[Dict[str, float]]:
        output = output.detach().cpu()
        return [{"mos": float(x[0])} for x in output]

    # ---------------------- PROPERTY ----------------------

    @property
    def sample_rate(self) -> int:
        return self.target_sr





