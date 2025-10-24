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
import traceback # Импортируем traceback
import csv

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
    - Element-level throughput tracking (items/sec)
    - Dynamic result saving after each batch
    - Clear audio path to prediction mapping
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
        self.model_wrapper = model_wrapper
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.save_every_n_batches = save_every_n_batches
        self.experiment_name = experiment_name
        self.async_save = async_save
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage and tracking
        self.results: List[Dict[str, Any]] = []
        self.batch_count = 0
        self.total_processed = 0  # elements processed
        self.start_time: Optional[float] = None
        
        # Throughput tracking (cumulative items/sec after each batch)
        self.throughput_history: List[float] = []
        self.batch_times: List[float] = []
        
        # File paths (only JSON report)
        self.json_output_file = self.output_dir / f"{experiment_name}_results.json"
        self.csv_output_file = self.output_dir / f"{experiment_name}_results.csv"
        
    def run(self) -> List[Dict[str, Any]]:
        self.start_time = time.time()
        total_elements = len(self.dataloader.dataset) if hasattr(self.dataloader, 'dataset') else None
        total_batches = len(self.dataloader)
        
        console.print(Panel.fit(
            (f"Starting inference over [bold green]{total_elements}[/bold green] files\n" if total_elements is not None else "Starting inference\n") +
            f"Output directory: [cyan]{self.output_dir}[/cyan]\n"
            f"Save frequency: Every [yellow]{self.save_every_n_batches}[/yellow] batch(es)",
            title="Inference Engine",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True
        ) as progress:
            task = progress.add_task(
                "Processing files...",
                total=total_elements if total_elements is not None else total_batches
            )
            
            for batch_idx, batch in enumerate(self.dataloader):
                batch_start_time = time.time()
                try:
                    batch_results = self._process_batch(batch, batch_idx)
                    self.results.extend(batch_results)
                    
                    batch_time = time.time() - batch_start_time
                    self.batch_times.append(batch_time)
                    num_items = len(batch_results)
                    self.total_processed += num_items

                    elapsed = max(time.time() - self.start_time, 1e-9)
                    current_throughput = self.total_processed / elapsed
                    self.throughput_history.append(current_throughput)
                    
                    progress.update(
                        task,
                        advance=num_items if total_elements is not None else 1,
                        description=(
                            f"Processed: {self.total_processed}/" + (str(total_elements) if total_elements is not None else f"~{total_batches}b") +
                            f" | Throughput: {current_throughput:.1f} items/sec"
                        )
                    )
                    
                    if (batch_idx + 1) % self.save_every_n_batches == 0:
                        self._save_results()
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed to process batch {batch_idx}: {e}[/red]")
                    traceback.print_exc()
                    continue
        
        # Save final results
        self._save_results()
        
        # Display completion summary
        self._display_completion_summary()
        
        return self.results
    
    def _process_batch(self, batch: Dict[str, Any], batch_idx: int) -> List[Dict[str, Any]]:
        waveforms = batch["waveform"]
        audio_paths = batch["audio_path"]
        transcripts = batch.get("transcript", [None] * len(audio_paths))
        
        predictions = self.model_wrapper.predict(waveforms)
        
        batch_results: List[Dict[str, Any]] = []
        # Duration in seconds for each item (after any trimming/padding)
        sample_rate = getattr(self.model_wrapper, "sample_rate", None)
        item_duration_sec: Optional[float] = None
        if sample_rate:
            # waveforms shape is [B, C, T]
            item_duration_sec = float(waveforms.shape[2]) / float(sample_rate)

        for i, (audio_path, transcript, pred) in enumerate(zip(audio_paths, transcripts, predictions)):
            result = {
                "audio_file_path": audio_path,
                "predictions": pred,
                "batch_info": {
                    "batch_index": batch_idx,
                    "sample_index": i
                }
            }
            if item_duration_sec is not None:
                result["duration_seconds"] = item_duration_sec
            batch_results.append(result)
        
        return batch_results

    # --------- METRICS ---------
    def _build_metrics(self) -> Dict[str, Any]:
        elapsed_time = (time.time() - self.start_time) if self.start_time else 0.0
        total_items = len(self.results)
        avg_throughput = total_items / elapsed_time if elapsed_time > 0 else 0.0
        avg_time_per_item = (elapsed_time / total_items) if total_items > 0 else 0.0
        last_throughput = self.throughput_history[-1] if self.throughput_history else 0.0
        # Compute global RTF = total processing time / total audio duration
        total_audio_seconds = 0.0
        for r in self.results:
            dur = r.get("duration_seconds")
            if isinstance(dur, (int, float)):
                total_audio_seconds += float(dur)
        average_rtf = (elapsed_time / total_audio_seconds) if total_audio_seconds > 0 else 0.0
        return {
            "total_files_processed": total_items,
            "total_time_seconds": round(elapsed_time, 3),
            "average_throughput_items_per_second": round(avg_throughput, 3),
            "current_throughput_items_per_second": round(last_throughput, 3),
            "average_time_per_file_seconds": round(avg_time_per_item, 3),
            "average_rtf": round(average_rtf, 4),
            "total_audio_seconds": round(total_audio_seconds, 3),
            "batches": len(self.batch_times),
        }
    
    def _save_results(self):
        """Save current results in JSON with metrics and results arrays."""
        metrics = self._build_metrics()
        payload = {
            "metrics": metrics,
            "results": self.results,
        }
        if self.async_save:
            asyncio.run(self._async_save_json(payload))
        else:
            self._save_json(payload)
        # Save/update CSV alongside JSON
        self._save_csv()
        console.print(f"[green]💾 Saved {metrics['total_files_processed']} results to {self.json_output_file} and {self.csv_output_file}[/green]")
    
    def _save_json(self, payload: Dict[str, Any]):
        with open(self.json_output_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    
    async def _async_save_json(self, payload: Dict[str, Any]):
        async with aiofiles.open(self.json_output_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(payload, indent=2, ensure_ascii=False))

    def _save_csv(self):
        """Save per-file metrics to CSV: columns = audio_file_path + metric keys."""
        # Determine union of prediction keys across all results
        metric_keys: List[str] = []
        seen = set()
        for r in self.results:
            preds = r.get("predictions", {}) or {}
            if isinstance(preds, dict):
                for k in preds.keys():
                    if k not in seen:
                        seen.add(k)
                        metric_keys.append(k)
        # Stable order
        metric_keys.sort()
        header = ["audio_file_path"] + metric_keys
        # Write CSV
        with open(self.csv_output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in self.results:
                row = [r.get("audio_file_path", "")] 
                preds = r.get("predictions", {}) or {}
                for k in metric_keys:
                    row.append(preds.get(k, ""))
                writer.writerow(row)
    
    def _display_completion_summary(self):
        if not self.start_time:
            return
        
        elapsed_time = time.time() - self.start_time
        total_items = len(self.results)
        avg_throughput = total_items / elapsed_time if elapsed_time > 0 else 0
        avg_time_per_item = (elapsed_time / total_items) if total_items > 0 else 0
        
        summary_table = Table(title="Inference Completed!", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Files Processed", str(total_items))
        summary_table.add_row("Total Time", f"{elapsed_time:.2f} seconds")
        summary_table.add_row("Average Throughput", f"{avg_throughput:.2f} items/sec")
        summary_table.add_row("Average Time per File", f"{avg_time_per_item:.3f} seconds")
        # Also print RTF if durations were available
        total_audio_seconds = sum(float(r.get("duration_seconds", 0.0)) for r in self.results)
        average_rtf = (elapsed_time / total_audio_seconds) if total_audio_seconds > 0 else 0.0
        summary_table.add_row("Average RTF", f"{average_rtf:.4f}")
        
        console.print(summary_table)
        console.print()
        
        files_panel = Panel.fit(
            f"📄 JSON Results: [cyan]{self.json_output_file}[/cyan]\n📄 CSV Results:  [cyan]{self.csv_output_file}[/cyan]",
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
    from dataset.сompact_collate_fn import compact_collate_fn
    
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": compact_collate_fn
    }
    
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["persistent_workers"] = True
    
    return DataLoader(dataset, **dataloader_kwargs)
