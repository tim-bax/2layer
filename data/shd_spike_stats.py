#!/usr/bin/env python3
"""
Compute SHD spike statistics per channel.

Reports, for each split:
  - total spikes per channel over all samples
  - average spikes per channel per sample
  - aggregate summary (mean/std/min/max over channels)

Example:
  python data/shd_spike_stats.py --split train
  python data/shd_spike_stats.py --split both --top_k 20
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_shd_data


def _parse_args():
    p = argparse.ArgumentParser(description="Compute average spikes per SHD channel")
    p.add_argument("--split", choices=["train", "test", "both"], default="train")
    p.add_argument("--data_path", type=str, default=None, help="SHD data path (auto-detect if omitted)")
    p.add_argument("--top_k", type=int, default=10, help="Print top-K most active channels")
    p.add_argument("--bottom_k", type=int, default=10, help="Print bottom-K least active channels")
    p.add_argument("--save_csv", type=str, default=None, help="Optional output CSV path")
    return p.parse_args()


def _channel_counts(samples: List[Tuple[List[np.ndarray], int]], n_units: int = 700) -> np.ndarray:
    counts = np.zeros((len(samples), n_units), dtype=np.int64)
    for i, (sample_spikes, _) in enumerate(samples):
        # sample_spikes is list length n_units; each entry is array of spike times for that unit.
        counts[i, :] = np.array([len(unit_spikes) for unit_spikes in sample_spikes], dtype=np.int64)
    return counts


def _print_split_stats(name: str, samples: List[Tuple[List[np.ndarray], int]], top_k: int, bottom_k: int):
    if not samples:
        print(f"\n{name}: no samples")
        return None

    counts = _channel_counts(samples)
    per_channel_total = counts.sum(axis=0)
    per_channel_avg = counts.mean(axis=0)  # avg spikes per channel per sample

    print(f"\n{name}")
    print("-" * 72)
    print(f"n_samples: {counts.shape[0]}")
    print(f"n_channels: {counts.shape[1]}")
    print(f"total_spikes_dataset: {int(per_channel_total.sum())}")
    print(f"avg_spikes_per_sample_total: {float(counts.sum(axis=1).mean()):.4f}")
    print(f"avg_spikes_per_channel_per_sample (mean over channels): {float(per_channel_avg.mean()):.6f}")
    print(
        "per-channel average spikes/sample: "
        f"min={float(per_channel_avg.min()):.6f} "
        f"max={float(per_channel_avg.max()):.6f} "
        f"std={float(per_channel_avg.std()):.6f}"
    )

    k = max(1, min(top_k, counts.shape[1]))
    top_idx = np.argsort(-per_channel_avg)[:k]
    print(f"top_{k}_channels_by_avg_spikes_per_sample:")
    for rank, ch in enumerate(top_idx, start=1):
        print(
            f"  {rank:2d}. channel={int(ch):3d} "
            f"avg={float(per_channel_avg[ch]):.6f} "
            f"total={int(per_channel_total[ch])}"
        )

    kb = max(1, min(bottom_k, counts.shape[1]))
    bottom_idx = np.argsort(per_channel_avg)[:kb]
    print(f"bottom_{kb}_channels_by_avg_spikes_per_sample:")
    for rank, ch in enumerate(bottom_idx, start=1):
        print(
            f"  {rank:2d}. channel={int(ch):3d} "
            f"avg={float(per_channel_avg[ch]):.6f} "
            f"total={int(per_channel_total[ch])}"
        )

    return per_channel_total, per_channel_avg


def _save_csv(path: str, split_name: str, per_channel_total: np.ndarray, per_channel_avg: np.ndarray):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "channel", "total_spikes", "avg_spikes_per_sample"])
        for ch in range(len(per_channel_total)):
            w.writerow([split_name, ch, int(per_channel_total[ch]), float(per_channel_avg[ch])])
    print(f"\nSaved CSV: {path}")


def main():
    args = _parse_args()
    train_data, test_data = load_shd_data(data_path=args.data_path)

    if args.split in ("train", "both"):
        train_stats = _print_split_stats("TRAIN", train_data, args.top_k, args.bottom_k)
        if args.save_csv and args.split == "train" and train_stats is not None:
            _save_csv(args.save_csv, "train", train_stats[0], train_stats[1])

    if args.split in ("test", "both"):
        test_stats = _print_split_stats("TEST", test_data, args.top_k, args.bottom_k)
        if args.save_csv and args.split == "test" and test_stats is not None:
            _save_csv(args.save_csv, "test", test_stats[0], test_stats[1])

    if args.split == "both" and args.save_csv:
        print("\n--save_csv with --split both is ambiguous for a single file.")
        print("Use --split train or --split test, or run twice with different CSV paths.")


if __name__ == "__main__":
    main()
