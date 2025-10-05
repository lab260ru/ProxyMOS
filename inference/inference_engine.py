import torch
import json
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader
import time
from datetime import datetime
import threading
from queue import Queue

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

from models.base_model import BaseModelWrapper

# Initialize Rich console
console = Console()


class InferenceEngine:
    """
    Engine for running audio quality assessment inference with batch processing,
    dynamic saving, and throughput monitoring.
    
    Features:
    - Batch processing with progress tracking
    - Dynamic result saving after each batch
    - Clear audio path to prediction mapping
    - Throughput calculation and monitoring
    - Beautiful Rich console output
    """
    
    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        dataloader: DataLoader,
        output_dir: str,
        save_every_n_batches: int = 1,  # Save after each batch by default
        experiment_name: str = "inference",
        async_save: bool = True
    ):
        """
        Initialize inference engine.
        
        Args:
            model_wrapper: Model wrapper instance
            dataloader: DataLoader for audio data
            output_dir: Directory to save results
            save_every_n_batches: Save results every N batches (default: 1)
            experiment_name: Name for experiment files
            async_save: Whether to use asynchronous saving
        """
        self.model_wrapper = model_wrapper
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.save_every_n_batches = save_every_n_batches
        self.experiment_name = experiment_name
        self.async_save = async_save
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage and tracking
        self.results = []
        self.batch_count = 0
        self.total_processed = 0
        self.start_time = None
        
        # Throughput tracking
        self.throughput_history = []
        self.batch_times = []
        
        # File paths for different output formats
        self.json_output_file = self.output_dir / f"{experiment_name}_results.json"
        self.csv_output_file = self.output_dir / f"{experiment_name}_results.csv"
        self.summary_file = self.output_dir / f"{experiment_name}_summary.json"
        
    def run(self) -> List[Dict[str, Any]]:
        """
        Run inference on all batches with beautiful progress tracking and throughput monitoring.
        
        Returns:
            List of all inference results
        """
        self.start_time = time.time()
        total_batches = len(self.dataloader)
        
        # Display start information
        console.print(Panel.fit(
            f"Starting inference with [bold green]{total_batches}[/bold green] batches\n"
            f"Output directory: [cyan]{self.output_dir}[/cyan]\n"
            f"Save frequency: Every [yellow]{self.save_every_n_batches}[/yellow] batch(es)",
            title="Inference Engine",
            border_style="blue"
        ))
        
        # Create progress bar with Rich
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True
        ) as progress:
            
            task = progress.add_task("Processing batches...", total=total_batches)
            
            for batch_idx, batch in enumerate(self.dataloader):
                batch_start_time = time.time()
                
                try:
                    # Run inference on batch
                    batch_results = self._process_batch(batch, batch_idx)
                    self.results.extend(batch_results)
                    
                    # Calculate batch processing time
                    batch_time = time.time() - batch_start_time
                    self.batch_times.append(batch_time)
                    
                    # Calculate throughput
                    current_throughput = len(batch_results) / batch_time if batch_time > 0 else 0
                    self.throughput_history.append(current_throughput)
                    
                    # Update progress with detailed information
                    progress.update(
                        task,
                        advance=1,
                        description=f"Batch {batch_idx + 1}/{total_batches} | "
                                   f"Processed: {len(self.results)} files | "
                                   f"Throughput: {current_throughput:.1f} files/sec"
                    )
                    
                    # Save intermediate results
                    if (batch_idx + 1) % self.save_every_n_batches == 0:
                        self._save_results()
                        self._save_summary()
                        
                except Exception as e:
                    console.print(f"[red]❌ Failed to process batch {batch_idx}: {e}[/red]")
                    continue
        
        # Save final results and summary
        self._save_results()
        self._save_summary()
        
        # Display completion summary
        self._display_completion_summary()
        
        return self.results
    
    def _process_batch(self, batch: Dict[str, Any], batch_idx: int) -> List[Dict[str, Any]]:
        """
        Process a single batch of audio data with clear path-to-prediction mapping.
        
        Args:
            batch: Batch data from DataLoader
            batch_idx: Batch index for logging
            
        Returns:
            List of results for this batch with clear audio path to prediction mapping
        """
        waveforms = batch["waveform"]
        audio_paths = batch["audio_path"]
        transcripts = batch.get("transcript", [None] * len(audio_paths))
        
        # Run model inference
        predictions = self.model_wrapper.predict(waveforms)
        
        # Combine results with clear mapping
        batch_results = []
        for i, (audio_path, transcript, pred) in enumerate(zip(audio_paths, transcripts, predictions)):
            # Create clear mapping structure
            result = {
                "audio_file_path": audio_path,  # Clear audio file path
                "transcript": transcript,
                "predictions": pred,  # Model predictions
                "batch_info": {
                    "batch_index": batch_idx,
                    "sample_index": i,
                    "timestamp": datetime.now().isoformat()
                }
            }
            batch_results.append(result)
        
        return batch_results
    
    def _save_results(self):
        """Save current results to multiple formats (JSON and CSV)."""
        if not self.results:
            return
        
        # Save JSON format
        if self.async_save:
            asyncio.run(self._async_save_json())
        else:
            self._save_json()
        
        # Save CSV format for easy analysis
        self._save_csv()
        
        console.print(f"[green]💾 Saved {len(self.results)} results to {self.output_dir}[/green]")
    
    def _save_json(self):
        """Save results in JSON format."""
        with open(self.json_output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    async def _async_save_json(self):
        """Asynchronously save results in JSON format."""
        async with aiofiles.open(self.json_output_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(self.results, indent=2, ensure_ascii=False))
    
    def _save_csv(self):
        """Save results in CSV format for easy analysis."""
        import csv
        
        if not self.results:
            return
        
        # Get all possible prediction keys
        all_pred_keys = set()
        for result in self.results:
            if 'predictions' in result and result['predictions']:
                all_pred_keys.update(result['predictions'].keys())
        
        # Write CSV
        with open(self.csv_output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['audio_file_path', 'transcript'] + list(all_pred_keys) + ['batch_index', 'sample_index', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                row = {
                    'audio_file_path': result['audio_file_path'],
                    'transcript': result.get('transcript', ''),
                    'batch_index': result['batch_info']['batch_index'],
                    'sample_index': result['batch_info']['sample_index'],
                    'timestamp': result['batch_info']['timestamp']
                }
                
                # Add prediction values
                if 'predictions' in result and result['predictions']:
                    row.update(result['predictions'])
                
                writer.writerow(row)
    
    def _save_summary(self):
        """Save inference summary with throughput statistics."""
        if not self.start_time:
            return
        
        elapsed_time = time.time() - self.start_time
        avg_throughput = len(self.results) / elapsed_time if elapsed_time > 0 else 0
        avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        
        summary = {
            "experiment_name": self.experiment_name,
            "total_files_processed": len(self.results),
            "total_batches": len(self.batch_times),
            "total_time_seconds": elapsed_time,
            "average_throughput_files_per_second": avg_throughput,
            "average_batch_time_seconds": avg_batch_time,
            "min_throughput": min(self.throughput_history) if self.throughput_history else 0,
            "max_throughput": max(self.throughput_history) if self.throughput_history else 0,
            "model_name": self.model_wrapper.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _display_completion_summary(self):
        """Display beautiful completion summary with throughput statistics."""
        if not self.start_time:
            return
        
        elapsed_time = time.time() - self.start_time
        avg_throughput = len(self.results) / elapsed_time if elapsed_time > 0 else 0
        
        # Create summary table
        summary_table = Table(title="Inference Completed!", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Files Processed", str(len(self.results)))
        summary_table.add_row("Total Batches", str(len(self.batch_times)))
        summary_table.add_row("Total Time", f"{elapsed_time:.2f} seconds")
        summary_table.add_row("Average Throughput", f"{avg_throughput:.2f} files/sec")
        summary_table.add_row("Average Time per File", f"{elapsed_time/len(self.results):.3f} seconds")
        
        if self.throughput_history:
            summary_table.add_row("Min Throughput", f"{min(self.throughput_history):.2f} files/sec")
            summary_table.add_row("Max Throughput", f"{max(self.throughput_history):.2f} files/sec")
        
        # Display results
        console.print(summary_table)
        console.print()
        
        # Display file locations
        files_panel = Panel.fit(
            f"📄 JSON Results: [cyan]{self.json_output_file}[/cyan]\n"
            f"📊 CSV Results: [cyan]{self.csv_output_file}[/cyan]\n"
            f"📋 Summary: [cyan]{self.summary_file}[/cyan]",
            title="Output Files",
            border_style="green"
        )
        console.print(files_panel)


def create_optimized_dataloader(
    dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create optimized DataLoader for audio inference.
    
    Args:
        dataset: Audio dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        prefetch_factor: Number of batches to prefetch
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Optimized DataLoader
    """
    # Use custom collate function to handle None values
    from dataset.сompact_collate_fn import compact_collate_fn
    
    # Only use prefetch_factor if num_workers > 0
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,  # No shuffling for inference
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": compact_collate_fn  # Use custom collate function
    }
    
    # Add multiprocessing-specific options only if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["persistent_workers"] = True
    
    return DataLoader(dataset, **dataloader_kwargs)
