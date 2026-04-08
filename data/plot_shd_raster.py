#!/usr/bin/env python3
"""
Plot a spike raster for one SHD sample and print its label.

Example:
  python data/plot_shd_raster.py --split train --index 0 --out shd_sample0.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_shd_data


def _parse_args():
    p = argparse.ArgumentParser(description="Plot SHD spike raster for one sample")
    p.add_argument("--split", choices=["train", "test"], default="train", help="Dataset split")
    p.add_argument("--index", type=int, default=0, help="Sample index within selected split")
    p.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="SHD dataset/cache path (default: auto-detect in loader)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="shd_raster.png",
        help="Output image path",
    )
    p.add_argument(
        "--max_time",
        type=int,
        default=None,
        help="Optional x-axis limit in ms (e.g., 700). Default: full sample duration",
    )
    return p.parse_args()


def _load_split(data_path, split):
    train_data, test_data = load_shd_data(
        data_path=data_path,
        train_samples_per_class=None,
        test_samples_per_class=None,
        target_classes=list(range(20)),
    )
    return train_data if split == "train" else test_data


def _plot_raster(sample_spikes, label, out_path, split, index, max_time=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    spike_times = []
    spike_units = []
    for unit_idx, unit_spikes in enumerate(sample_spikes):
        if len(unit_spikes) == 0:
            continue
        unit_spikes = np.asarray(unit_spikes)
        if max_time is not None:
            unit_spikes = unit_spikes[unit_spikes <= max_time]
        spike_times.extend(unit_spikes.tolist())
        spike_units.extend([unit_idx] * len(unit_spikes))

    ax.scatter(spike_times, spike_units, s=4, marker="|")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Unit index")
    ax.set_title(f"SHD raster | split={split} index={index} label={label}")
    ax.set_ylim(-1, len(sample_spikes))
    if max_time is not None:
        ax.set_xlim(0, max_time)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = _parse_args()
    data = _load_split(args.data_path, args.split)

    if len(data) == 0:
        raise RuntimeError(f"No samples found in split '{args.split}'.")
    if args.index < 0 or args.index >= len(data):
        raise IndexError(f"Index {args.index} out of range for split '{args.split}' (size={len(data)}).")

    sample_spikes, label = data[args.index]
    _plot_raster(
        sample_spikes=sample_spikes,
        label=label,
        out_path=args.out,
        split=args.split,
        index=args.index,
        max_time=args.max_time,
    )

    n_events = int(sum(len(u) for u in sample_spikes))
    print(f"Saved raster to: {args.out}")
    print(f"Sample: split={args.split}, index={args.index}, label={label}, n_events={n_events}")


if __name__ == "__main__":
    main()
