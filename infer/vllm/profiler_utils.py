#!/usr/bin/env python3

"""
Pytorch profiler utilities for FMWork Workload Characterization.

Profiling and memory monitoring capabilities for characterizing LLM inference workloads on Spyre Accelerators
"""

import os
import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

class FMWorkProfiler:
    """ 
    Wrapper for PyTorch profiler with FMWork Integration. 
    """
    def __init__(
        self,
        output_dir: str,
        enabled: bool = True,
        wait_steps: int = 1,
        warmup_steps: int = 1,
        active_steps: int = 3,
        profile_memory: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        record_shapes: bool = True
    ):
        """
        Initializes the FMWorkProfiler.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

        self.config = {
            'wait': wait_steps,
            'warmup': warmup_steps,
            'active': active_steps,
            'profile_memory': profile_memory,
            'with_stack': with_stack,
            'with_flops': with_flops,
            'record_shapes': record_shapes
        }

        self.profiler = None
        self.step_count = 0

    def start(self):

        if not self.enabled:
            return
        activities = [
            torch.profiler.ProifilerActivity.CPU,
        ]

        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.profiler = torch.profiler.profile(
            activities = activities,
            schedule = torch.profiler.schedule(
                wait - self.config['wait'],
                warmup = self.config['warmup'],
                active = self.config['active'],
                repeat = 1
            ),
            on_trace_ready = torch.profilers.tensorboard_trace_handler(
                str(self.output_dir)
            ),
            record_shapes = self.config['record_shapes'],
            profile_memory = self.config['profile_memory'],
            with_flops = self.config['with_flops'],
            with_stack = self.config['with_stack']
        )
        self.profiler.__enter__()
        print(f"[PROFILER] Started - Ouput: {self.output_dir}")

    def step(self):
        if self.enabled and self.profiler:
            self.profiler.step()
            self.step_count += 1

    def stop(self):
        if self.enabled and self.profiler:
            self.profiler.__exit__(None, None, None)
            print(f"[PROFILER] Stopped - Ouput: {self.output_dir}")
            self._save_metadata()

    def _save_metadata(self):
        metadata = {
            'total_steps' : self.step_count,
            'config' : self.config,
            'output_dir' : str(self.output_dir)
        }

class MemoryMonitor:

    def __init__(self, outout_dir: str):
        self.output_dir = outout_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots = []
    
    def snapshot(self, label: str = "") -> Dict[str, Any]:
        snapshot = {
            'label' : label, 
            'host' : self._get_host_memory(), 
            'devices' : self._get_device_memory()
        }
        self.snapshots.append(snapshot)
        return snapshot

    def _get_host_memory(self) -> Dict[str, Any]:
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'total_gb' : mem.total / (1024**3),
                'available_gb' : mem.available / (1024**3),
                'used_gb' : mem.used / (1024**3),
                'percent' : mem.percent
            }
        except ImportError:
            return {'error' : 'psutil not installed'}
    
    def _get_device_memory(self) -> List[Dict[str, Any]]:
        device = []
        devices = []
        if torch.cuda.is_available():
            for i in range (torch.cude.device_count()):
                devices.append({
                    'device_id' : i,
                    'name' : torch.cuda.get_device_name(i),
                    'allocated_gb' : torch.cuda.memory_allocated(i) / (1024**3),
                    'reserved_gb' : torch.cuda.memory_reserved(i) / (1024**3),
                    'max_allocatd_gb' : torch.cuda.max_memory_allocated(i) / (1024**3)
                })
        return devices

    def save(self):
        output_path = self.output_dir / 'memory_snapshots.json'
        with open(output_path, 'w') as f:
            json.dump(self.snapshots, f, indent = 2)
        print(f"[MEMORY] Saved {len(self.snapshots)} snapshots to {output_path}")
        