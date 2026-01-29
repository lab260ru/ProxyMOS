import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from proxymostrainer import OmniMOS, ProxyMOSTrainer
from dataset.audio_dataset import AudioDataset
from utils.utils import (
    create_argument_parser,
    load_config,
    merge_config_with_args,
    setup_device,
    print_gpu_info,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


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

    from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
    
    if accelerator.is_main_process:
        ckpt = torch.load("omniASR-W2V-300M.pt", map_location="cpu")
    else:
        ckpt = None

    ckpt = accelerator.broadcast_object_list([ckpt])[0]
    if accelerator.is_main_process:
        print("Creating model config...")

    model_cfg = Wav2Vec2Config(
        extractor_mode="layer_norm",
        conv_feature_layers="[(512,10,5),(512,3,2),(512,3,2),(512,3,2),(512,2,2),(512,2,2),(512,2,2)]",
        encoder_layers=24,
        encoder_embed_dim=1024,
        encoder_ffn_embed_dim=4096,
        encoder_attention_heads=16,
        activation_fn="gelu",
        layer_norm_first=True,
        final_dim=768,  
    )
    if accelerator.is_main_process:
        print("Building model...")
    encoder = Wav2Vec2Model(model_cfg)
    if accelerator.is_main_process:
        print("Loading pretrained weights...")
    missing, unexpected = encoder.load_state_dict(ckpt["model"], strict=False)
    print(f"Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    encoder.train()
    for p in encoder.parameters():
        p.requires_grad = True
    if accelerator.is_main_process:
        print("Wrapping in OmniMOS...")
    model = OmniMOS(encoder)
    if accelerator.is_main_process:
        print("Loading datasets...")
    train_ds = AudioDataset(csv_file=config["data"]["train_csv"], time_length=10)
    val_ds = AudioDataset(csv_file=config["data"]["val_csv"], time_length=10)

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