from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, List


class BaseModelWrapper(ABC):
    """
    Abstract base class for all audio quality assessment model wrappers.
    
    Provides unified interface:
    - All methods work with torch.Tensor
    - Standard pipeline: preprocess -> forward -> postprocess
    - Device management and configuration handling
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        """
        Initialize model wrapper.
        
        Args:
            config: Model configuration dictionary
            device: Device to run model on (default: "cuda")
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
    
    @abstractmethod
    def load_model(self):
        """
        Load model and move to target device.
        
        Should set self.model attribute.
        """
        raise NotImplementedError
    
    @abstractmethod
    def preprocess(self, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Preprocess audio batch for model input.
        
        Args:
            audio_batch: Input audio tensor [batch, channels, samples]
            
        Returns:
            Preprocessed audio tensor [batch, channels, samples]
        """
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Run model forward pass.
        
        Args:
            audio: Preprocessed audio tensor [batch, channels, samples]
            
        Returns:
            Model output tensor [batch, n_metrics]
        """
        raise NotImplementedError
    
    @abstractmethod
    def postprocess(self, output: torch.Tensor) -> List[Dict[str, float]]:
        """
        Postprocess model output to readable format.
        
        Args:
            output: Model output tensor [batch, n_metrics]
            
        Returns:
            List of dictionaries with metric names and values
        """
        raise NotImplementedError
    
    def predict(self, audio_batch: torch.Tensor) -> List[Dict[str, float]]:
        """
        Run complete prediction pipeline.
        
        Args:
            audio_batch: Input audio tensor [batch, channels, samples]
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            self.load_model()
        
        audio_batch = audio_batch.to(self.device)
        
        with torch.no_grad():
            preprocessed = self.preprocess(audio_batch, self.sample_rate)
            output = self.forward(preprocessed)
            results = self.postprocess(output)
        
        return results
    