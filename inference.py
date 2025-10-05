import sys
from dataset.audio_dataset import AudioDataset
from inference.inference_engine import InferenceEngine, create_optimized_dataloader
from utils.utils import (
    get_model_wrapper,
    create_argument_parser,
    load_config,
    merge_config_with_args,
    setup_device,
    validate_paths,
    print_config_summary,
    print_gpu_info,
    console
)


def main():
    """
    Main function for audio quality assessment inference.
    
    This function orchestrates the entire inference pipeline:
    1. Parse command line arguments
    2. Load and merge configuration
    3. Validate paths and setup devicecd MOS_research
    4. Create dataset and dataloader
    5. Initialize model wrapper
    6. Run inference engine
    """
    
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load and merge configuration
    config = load_config(args.config)
    config = merge_config_with_args(config, args)
    
    # Validation and setup
    validate_paths(config)
    config['model']['device'] = setup_device(config['model']['device'])
    
    # Print configuration summary
    print_config_summary(config)
    print_gpu_info()
    
    try:
        # Create dataset
        dataset = AudioDataset(
            manifest=config['dataset'].get('manifest', []),  # Use manifest if available
            sample_rate=config['dataset'].get('sample_rate', 16000),
            file_format=config['dataset'].get('file_extension', '.wav'),
            time_length=config['dataset'].get('max_length', 4)
        )
        console.print(f"[green]✅ Loaded {len(dataset)} audio files[/green]\n")
        
        # Create DataLoader
        dataloader = create_optimized_dataloader(
            dataset,
            batch_size=config['inference']['batch_size'],
            num_workers=config['inference']['num_workers'],
            prefetch_factor=config['inference'].get('prefetch_factor', 2)
        )
        
        # Create model wrapper
        model_config = config['model']
        model_specific_config = model_config.get(model_config['name'], {})
        model_specific_config.update({
        'checkpoint_path': model_specific_config.get('checkpoint_path') or model_config.get('checkpoint_path'),
        })
        model_wrapper = get_model_wrapper(
            model_name=model_config['name'],
            config=model_specific_config,
            device=model_config['device']
        )
        
        # Create inference engine
        engine = InferenceEngine(
            model_wrapper=model_wrapper,
            dataloader=dataloader,
            output_dir=config['inference']['output_dir'],
            save_every_n_batches=config['inference'].get('save_every_n_batches', 10),
            experiment_name=args.experiment_name,
            async_save=not args.no_async_save
        )
        
        # Run inference
        results = engine.run()
        
        console.print(f"\n[bold green]🎉 SUCCESS! Inference completed! Processed {len(results)} files[/bold green]\n")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Inference interrupted by user[/yellow]")
        sys.exit(0)
        
    except Exception as e:
        console.print(f"\n[red]❌ ERROR: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()