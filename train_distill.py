import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import Wav2Vec2Model,  Wav2Vec2FeatureExtractor
import pandas as pd
import numpy as np

from proxymostrainer import OmniMOS, ProxyMOSTrainer
from dataset.audio_dataset import AudioDataset
from utils.utils import (
    create_argument_parser,
    load_config,
    merge_config_with_args,
    setup_device,
    print_gpu_info,
)
from pathlib import Path
import warnings
from encoders import build_encoder
warnings.filterwarnings("ignore")




torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.deterministic = True

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_checkpoint(checkpoint_path: str):
        """Загрузить fairseq2 веса из .pt файла"""
        import omnilingual_asr
        from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub

        hub = get_wav2vec2_model_hub()
        fs2_config = hub.get_model_config('omniASR_W2V_300M')
        model = hub.load_custom_model(
            Path(checkpoint_path),
            config=fs2_config,
            device=torch.device("cpu"),
        )
        return model,  fs2_config
def load_checkpoint_hf(checkpoint_path: str):   
    hf_model =   Wav2Vec2Model.from_pretrained(checkpoint_path, trust_remote_code=True , token = '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')
    hf_model.eval()
    return  hf_model

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    config = merge_config_with_args(config, args)

    config["model"]["device"] = setup_device(config["model"].get("device", "cuda"))
    print_gpu_info()

    accelerator = Accelerator(
        mixed_precision=config["accelerate"]["mixed_precision"],
    )

    if accelerator.is_main_process:
        print("Loading Omni-asr.")
    
    if  config["model"]["type"] == "fairseq2":
        encoder, config_omni = load_checkpoint(
            "omniASR-W2V-300M.pt",  # 300M параметров
        )
        encoder = build_encoder("fairseq2", encoder)
    else :
         encoder = load_checkpoint_hf('converted12')
         encoder = build_encoder("huggingface", encoder)
         
    
    
    if accelerator.is_main_process and config["model"]["type"] == "fairseq2":
        print("✅ Model loaded successfully!")
        print(f"Model config: {config_omni}")

    # Разморозить все слои
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad = True
    
    if accelerator.is_main_process:
        print("Wrapping in OmniMOS...")
    
    model = OmniMOS(encoder=encoder)
    
    if config["accelerate"]["mixed_precision"] == "bf16":
     model = model.to(torch.bfloat16)
    
    
    if accelerator.is_main_process:
        print("Loading datasets...")
    
    train_ds = AudioDataset(
        csv_file=config["data"]["train_csv"], 
        segment_seconds=8  
    )
    val_ds = AudioDataset(
        csv_file=config["data"]["val_csv"], 
        segment_seconds=8
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    
    if accelerator.is_main_process:
        print("Creating trainer...")
    
    trainer = ProxyMOSTrainer(
        accelerator=accelerator,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        max_grad_norm=float(config["training"]["max_grad_norm"]),
        save_dir=config["output"]["save_dir"],
    )
    
    if accelerator.is_main_process:
        print("Starting training...")
    
    trainer.train(
        epochs=config["training"]["epochs"],
        early_stop_patience=config["training"]["early_stop_patience"],
    )


if __name__ == "__main__":
    main()