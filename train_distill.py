# train_proxymos.py
import torch
import json
from pathlib import Path
from dataset.audio_dataset import AudioDataset
from proxymostrainer import ProxyMOSTrainer
from utils.utils import (
    create_argument_parser,
    load_config,
    merge_config_with_args,
    setup_device,
    validate_paths,
    print_gpu_info,
)
import distillmos

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load and merge configuration
    config = load_config(args.config)
    config = merge_config_with_args(config, args)
    

    validate_paths(config)
    config['model']['device'] = setup_device(config['model']['device'])
    

    print_gpu_info()

   
    
    # Создаем datasets напрямую с manifest
    train_dataset = AudioDataset(
        csv_file='pathtocsv',
    )
    
    val_dataset = AudioDataset(
        csv_file='pathtocsv',
    )
    

    sqa_model = distillmos.ConvTransformerSQAModel()
    

    trainer = ProxyMOSTrainer(
        model= sqa_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        device=config['training']['device'],
    )
    
    trainer.train(epochs=config['training']['epochs'])


if __name__ == "__main__":
    main()