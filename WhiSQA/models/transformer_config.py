from enum import Enum
import torch
from torch.nn.functional import pad

class Input(Enum):
    MFCC = 0
    XLSR = 1

class CenterCrop(torch.nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor):
        # Center crop.
        unsqueezed = False
        if x.dim() == 2:
            unsqueezed = True
            x = x.unsqueeze(0)
        assert x.dim() == 3 # N, L, C

        if x.size(1) > self.seq_len:
            center_start_idx = int(x.size(1) / 2 - self.seq_len / 2)
            start_idx = center_start_idx
            end_idx = start_idx + self.seq_len
            x = x[:, start_idx:end_idx, :]
        if x.size(1) < self.seq_len:
            to_pad = self.seq_len - x.size(1)
            # Pad the end of sequence dimension.
            x = pad(x, (0,0,0,to_pad,0,0), mode="constant", value=0.0)

        if unsqueezed:
            x = x.squeeze(0)

        return x
    
class Config:

    name: str = None
    input: Input = None
    feat_seq_len: int = None
    dim_input: int = None
    dim_transformer: int = None
    dim_head_in: int = None
    dim_head_out: int = None

    def __init__(
        self,
        name: str,
        input: Input,
        feat_seq_len: int,
        dim_transformer: int = None,
        xlsr_name: str = None,
        nhead_transformer: int = 4,
        nlayers_transformer: int = 2,
    ):
        if input == Input.MFCC:
            xlsr_name = None

        # Check valid parameters.
        assert feat_seq_len > 0, "feat_seq_len must be positive."

        # Save parameters.
        self.name = name
        self.input = input
        self.feat_seq_len = feat_seq_len
        self.dim_transformer = dim_transformer
        self.xlsr_name = xlsr_name
        self.nhead_transformer = nhead_transformer
        self.nlayers_transformer = nlayers_transformer
        if xlsr_name is not None:
            # From XLS-R paper Table 2: Model architectures.
            if xlsr_name == "wav2vec2-xls-r-300m":
                _b = 24
                _h = 1024
            elif xlsr_name == "wav2vec2-xls-r-1b":
                _b = 48
                _h = 1280
            elif xlsr_name == "wav2vec2-xls-r-2b":
                _b = 48
                _h = 1920
            elif xlsr_name == "hubert_encoder":
                _b = -1
                _h = 512
            elif xlsr_name == "hubert_encoder_t":
                _b = -1
                _h = 384
            elif xlsr_name == "hubert_full":
                _b = -1
                _h = 768
            elif xlsr_name == "hubert_full_t":
                _b = -1
                _h = 384
            elif xlsr_name == "whisper_encoder":
                _b = -1
                _h = 768
            elif xlsr_name == "whisper_encoder_ref":
                _b = -1
                _h = 768*2
            elif xlsr_name == "whisper_encoder_t":
                _b = -1
                _h = 1500
            elif xlsr_name == "whisper_full":
                _b = -1
                _h = 768
            elif xlsr_name == "whisper_full_t":
                _b = -1
                _h = 384
            self.xlsr_layers = _b + 1  # +1 for CNN activation "layer0"
            self.dim_input = _h
        else:
            if self.feat_seq_len == 80: #handle transposed mfcc
                self.xlsr_layers = None
                self.dim_input = 3000
            else:
                self.xlsr_layers = None
                self.dim_input = 80  # MFCC

        self.dim_head_in = self.dim_transformer  # * self.feat_seq_len
        self.dim_head_out = 1

        self.dropout = 0.1  # TODO

# Length of feature frame window.
FEAT_SEQ_LEN = 256

# Preset Config instances were removed as unused. Keep only API classes and FEAT_SEQ_LEN.