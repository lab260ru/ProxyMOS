# models/whisqa_wrapper.py
import torch
import torchaudio
from typing import Dict, List
from .base_model import BaseModelWrapper


class WhiSQAWrapper(BaseModelWrapper):
    """
    Wrapper for WhiSQA (Whisper-based Speech Quality Assessment).

    Features:
        - Predicts MOS (Mean Opinion Score) for speech quality
        - Based on Whisper encoder features
        - Supports single and multi-dimensional models
        - Automatic resampling to 16 kHz
    
    Source:
        https://github.com/leto19/WhiSQA
    """

    def load_model(self):
        """
        Load WhiSQA model from checkpoint.
        
        Config options:
            - checkpoint_path: Path to model checkpoint file (required)
            - model_type: "single" or "multi" (default: "single")
            - device: "cpu" or "cuda"
        """
        from WhiSQA.models.whisper_ni_predictors import (
            whisperMetricPredictorEncoderLayersTransformerSmall,
            whisperMetricPredictorEncoderLayersTransformerSmalldim,
        )

        checkpoint_path = self.config.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("'checkpoint_path' is required in config to load WhiSQA model")

        model_type = self.config.get("model_type", "multi")
        print(f"[WhiSQA] 🔄 Loading model type '{model_type}' from {checkpoint_path}")

        # Select model architecture
        if model_type == "single":
            model = whisperMetricPredictorEncoderLayersTransformerSmall()
        elif model_type == "multi":
            model = whisperMetricPredictorEncoderLayersTransformerSmalldim()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'single' or 'multi'")

        # Load checkpoint
        state = torch.load(checkpoint_path, map_location=self.device)

        # Extract model state dict if wrapped
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        model.load_state_dict(state)
        model.to(self.device)
        model.eval()

        self.model = model
        print(f"[WhiSQA] ✅ Model successfully loaded on device: {self.device}\n")

    # ---------------------- PREPROCESS ----------------------

    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Preprocess audio for WhiSQA model.
        
        Steps:
            1. Convert stereo to mono
            2. Resample to 16 kHz (WhiSQA requirement)
            3. Normalize amplitude to [-1, 1]
        
        Args:
            audio_batch: Input tensor [batch, channels, samples]
            sample_rate: Original audio sample rate
            
        Returns:
            torch.Tensor: Preprocessed audio [batch, 1, samples]
        """
        # Convert stereo to mono
        if audio_batch.size(1) > 1:
            print("[WhiSQA] ℹ Converting Stereo → Mono")
            audio_batch = audio_batch.mean(dim=1, keepdim=True)

        # Resample to 16 kHz if needed
        target_sr = 16000
        if sample_rate != target_sr:
            print(f"[WhiSQA] ⚠ Resampling: {sample_rate} Hz → {target_sr} Hz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=target_sr
            ).to(self.device)
            audio_batch = resampler(audio_batch)

        # Normalize to [-1, 1]
        max_val = audio_batch.abs().amax(dim=-1, keepdim=True)
        if torch.any(max_val > 0):
            audio_batch = audio_batch / (max_val + 1e-9)

        return audio_batch

    # ---------------------- INFERENCE ----------------------

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Run WhiSQA model inference.
        
        Args:
            audio: Preprocessed audio [batch, 1, samples]
            
        Returns:
            torch.Tensor: [batch, 1] MOS predictions
        """
        # Remove channel dimension: [batch, samples]
        audio_2d = audio.squeeze(1)

        with torch.no_grad():
            outputs = self.model(audio_2d)

        # Handle different output formats
        if isinstance(outputs, dict):
            mos = outputs.get("mos") or outputs.get("score") or outputs.get("prediction")
            if mos is None:
                raise ValueError(f"Unable to extract MOS from dict keys: {outputs.keys()}")
        elif isinstance(outputs, torch.Tensor):
            mos = outputs
        else:
            raise ValueError(f"Unexpected output type: {type(outputs)}")

        # Ensure output shape is [batch, 1]
        if mos.dim() == 1:
            mos = mos.unsqueeze(1)

        return mos

    # ---------------------- POSTPROCESS ----------------------

    def postprocess(self, output: torch.Tensor) -> List[Dict[str, float]]:
        """
        Convert model output to MOS metrics.
        
        Args:
            output: Model predictions [batch, 1]
            
        Returns:
            List[Dict[str, float]]: MOS quality scores
        """
        results = []
        for pred in output.cpu():
            mos_value = float(pred[0])
            results.append({
                "mos": mos_value,
                "quality_score": mos_value
            })
        return results

    # ---------------------- PROPERTY ----------------------

    @property
    def sample_rate(self) -> int:
        """WhiSQA requires 16 kHz audio."""
        return 16000




