"""
SHD count-binning preprocessing à la Bittar & Garner (sparch) and Fabre et al. 2025
(https://arxiv.org/abs/2506.06374).

Pipeline per sample:
  1. Read raw events (units, times in seconds) from the HDF5 file.
  2. Time binning:    bin index = floor(time_ms / bin_size_ms),
                      values in each (bin, channel) cell = number of spikes that
                      fell into that cell (a "count tensor", not 0/1 occupancy
                      and not an alpha-kernel current).
  3. Spatial binning: sum-pool every `collapse_factor` consecutive input
                      channels into one output channel (e.g. 700 -> 140 with
                      collapse_factor=5).
  4. Length handling: every sample is zero-padded out to the same number of
                      time bins so all examples have shape (T_bins, n_channels_out).

Tunable knobs:
  - bin_size_ms      : time bin width in ms          (paper default 4.0)
  - collapse_factor  : channel pool size             (paper default 5)
  - max_duration_ms  : fixed window length in ms     (paper default 1400.0)

Cross-references:
  - Paper § 4.1 Experimental setup
  - sparch dataloaders/spiking_datasets.py (BSD-3, Bittar & Garner)

This module is intentionally standalone (only NumPy + h5py/tables + stdlib) so
it doesn't pull in the TensorFlow dependency that `data/shd.py` carries.
"""
from __future__ import annotations

import gzip
import os
import shutil
import urllib.request
from typing import List, Optional, Tuple

import numpy as np


N_UNITS_SHD = 700
ZENKE_BASE_URL = "https://zenkelab.org/datasets"


def _open_hdf5(hdf5_path: str):
    """Open an HDF5 file with PyTables if available, else h5py.

    Returns (file_handle, units_dataset, times_dataset, labels_array, use_h5py_flag).
    Imports are deferred so the binning helpers can be used without hdf5 deps installed.
    """
    try:
        import tables  # noqa: WPS433
        fh = tables.open_file(hdf5_path, mode="r")
        return (
            fh,
            fh.root.spikes.units,
            fh.root.spikes.times,
            np.asarray(fh.root.labels, dtype=np.int64),
            False,
        )
    except (ImportError, ValueError):
        try:
            import h5py  # noqa: WPS433
        except ImportError as e:
            raise ImportError(
                "Either pytables or h5py must be installed to read SHD HDF5 files. "
                "pip install h5py"
            ) from e
        fh = h5py.File(hdf5_path, mode="r")
        return (
            fh,
            fh["spikes"]["units"],
            fh["spikes"]["times"],
            np.asarray(fh["labels"], dtype=np.int64),
            True,
        )


def _default_cache_dir() -> str:
    """Same priority order as data/shd.py for compatibility."""
    if "SHD_CACHE_DIR" in os.environ:
        return os.environ["SHD_CACHE_DIR"]
    if os.path.exists("/share/neurocomputation/Tim/SHD_data"):
        return "/share/neurocomputation/Tim/SHD_data"
    if "SCRATCH" in os.environ:
        return os.path.join(os.environ["SCRATCH"], "data")
    if "TMPDIR" in os.environ and os.environ.get("TMPDIR") != "/tmp":
        return os.path.join(os.environ["TMPDIR"], "data")
    if os.path.exists("/scratch"):
        return "/scratch/data"
    return os.path.expanduser("~/data")


def _download_and_gunzip(filename: str, cache_dir: Optional[str] = None) -> str:
    """Download `filename` from Zenke lab and gunzip it. Returns the .h5 path."""
    cache_dir = cache_dir or _default_cache_dir()
    cache_subdir = os.path.join(cache_dir, "hdspikes")
    os.makedirs(cache_subdir, exist_ok=True)

    gz_path = os.path.join(cache_subdir, filename)
    h5_path = gz_path[:-3] if gz_path.endswith(".gz") else gz_path + ".h5"

    if not os.path.isfile(gz_path) and not os.path.isfile(h5_path):
        url = f"{ZENKE_BASE_URL}/{filename}"
        print(f"Downloading {url} -> {gz_path}", flush=True)
        urllib.request.urlretrieve(url, gz_path)

    if not os.path.isfile(h5_path) or (
        os.path.isfile(gz_path) and os.path.getctime(gz_path) > os.path.getctime(h5_path)
    ):
        print(f"Decompressing {gz_path}", flush=True)
        with gzip.open(gz_path, "rb") as fin, open(h5_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return h5_path


def bin_shd_sample(
    units: np.ndarray,
    times_sec: np.ndarray,
    bin_size_ms: float = 4.0,
    collapse_factor: int = 5,
    n_units_in: int = N_UNITS_SHD,
    max_duration_ms: Optional[float] = 1400.0,
    binarize: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    """Bin one SHD sample's raw events into a (T_bins, n_channels_out) count matrix.

    Args:
        units:           int array of input channel indices in [0, n_units_in).
        times_sec:       float array of spike timestamps in seconds.
        bin_size_ms:     time bin width, ms.
        collapse_factor: number of consecutive input channels summed into one
                         output channel (1 = no collapsing).
        n_units_in:      number of original input channels (700 for SHD).
        max_duration_ms: events past this are dropped (not squashed). Sets
                         T_bins = ceil(max_duration_ms / bin_size_ms). If None,
                         T_bins is set to the sample's own max bin index + 1.
        binarize:        if True, clip counts to {0, 1} after binning.
        dtype:           output dtype.

    Returns:
        x: ndarray of shape (T_bins, n_channels_out).
    """
    units = np.asarray(units, dtype=np.int64)
    times_ms = np.asarray(times_sec, dtype=np.float64) * 1000.0

    if max_duration_ms is not None:
        keep = (times_ms >= 0.0) & (times_ms < max_duration_ms)
        times_ms = times_ms[keep]
        units = units[keep]
        T_bins = max(1, int(np.ceil(max_duration_ms / bin_size_ms)))
    else:
        keep = times_ms >= 0.0
        times_ms = times_ms[keep]
        units = units[keep]
        if times_ms.size:
            T_bins = int(np.floor(times_ms.max() / bin_size_ms)) + 1
        else:
            T_bins = 1

    n_channels_out = (n_units_in + collapse_factor - 1) // collapse_factor

    t_idx = (times_ms / bin_size_ms).astype(np.int64)
    np.minimum(t_idx, T_bins - 1, out=t_idx)
    c_idx = units // collapse_factor

    valid = (c_idx >= 0) & (c_idx < n_channels_out) & (t_idx >= 0)
    t_idx = t_idx[valid]
    c_idx = c_idx[valid]

    x = np.zeros((T_bins, n_channels_out), dtype=dtype)
    np.add.at(x, (t_idx, c_idx), 1)
    if binarize:
        np.minimum(x, 1, out=x)
    return x


class SHDBinnedLoader:
    """SHD loader that returns dense count-binned tensors.

    No TF dependency; uses urllib for download and tables/h5py for HDF5 access.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        bin_size_ms: float = 4.0,
        collapse_factor: int = 5,
        max_duration_ms: Optional[float] = 1400.0,
    ):
        if bin_size_ms <= 0:
            raise ValueError("bin_size_ms must be > 0")
        if collapse_factor < 1:
            raise ValueError("collapse_factor must be >= 1")

        self.data_path = data_path
        self.n_units = N_UNITS_SHD
        self.bin_size_ms = float(bin_size_ms)
        self.collapse_factor = int(collapse_factor)
        self.max_duration_ms = float(max_duration_ms) if max_duration_ms is not None else None
        self.n_channels_out = (self.n_units + self.collapse_factor - 1) // self.collapse_factor
        self.T_bins = (
            max(1, int(np.ceil(self.max_duration_ms / self.bin_size_ms)))
            if self.max_duration_ms is not None
            else None
        )

    def get_dataset_binned(
        self,
        split: str = "train",
        max_samples_per_class: Optional[int] = None,
        target_classes: Optional[List[int]] = None,
        binarize: bool = False,
        dtype=np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (X, y, lengths):
            X       : (N, T_bins, n_channels_out) float array
            y       : (N,)                         int64
            lengths : (N,)                         int64, last populated time bin + 1
                      (useful if you want to mask the trailing zeros)
        """
        if target_classes is None:
            target_classes = list(range(20))
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split: {split}")
        filename = f"shd_{split}.h5.gz"
        hdf5_path = _download_and_gunzip(filename, cache_dir=self.data_path)

        fh, units_ds, times_ds, labels, use_h5py = _open_hdf5(hdf5_path)

        xs: List[np.ndarray] = []
        ys: List[int] = []
        lens: List[int] = []
        class_counts = {c: 0 for c in target_classes}

        for i in range(labels.shape[0]):
            label = int(labels[i])
            if label not in target_classes:
                continue
            if max_samples_per_class is not None and class_counts[label] >= max_samples_per_class:
                continue
            u = units_ds[i][:] if use_h5py else units_ds[i]
            t = times_ds[i][:] if use_h5py else times_ds[i]

            x = bin_shd_sample(
                u, t,
                bin_size_ms=self.bin_size_ms,
                collapse_factor=self.collapse_factor,
                n_units_in=self.n_units,
                max_duration_ms=self.max_duration_ms,
                binarize=binarize,
                dtype=dtype,
            )
            populated = np.flatnonzero(x.any(axis=1))
            length = int(populated[-1] + 1) if populated.size else 0

            xs.append(x)
            ys.append(label)
            lens.append(length)
            class_counts[label] += 1

        fh.close()

        if self.T_bins is not None:
            X = np.stack(xs, axis=0)
        else:
            T_max = max(x.shape[0] for x in xs) if xs else 1
            X = np.zeros((len(xs), T_max, self.n_channels_out), dtype=dtype)
            for k, x in enumerate(xs):
                X[k, : x.shape[0]] = x

        y = np.asarray(ys, dtype=np.int64)
        lengths = np.asarray(lens, dtype=np.int64)

        print(
            f"[shd_binned/{split}] N={X.shape[0]}, "
            f"shape={tuple(X.shape)}, "
            f"bin={self.bin_size_ms}ms, collapse={self.collapse_factor} "
            f"(700 -> {self.n_channels_out} channels), "
            f"mean count/bin={X.mean():.4f}, max count/bin={X.max():.0f}, "
            f"median length={int(np.median(lengths))} bins",
            flush=True,
        )
        return X, y, lengths


def load_shd_binned(
    bin_size_ms: float = 4.0,
    collapse_factor: int = 5,
    max_duration_ms: Optional[float] = 1400.0,
    train_samples_per_class: Optional[int] = None,
    test_samples_per_class: Optional[int] = None,
    target_classes: Optional[List[int]] = None,
    binarize: bool = False,
    dtype=np.float32,
    data_path: Optional[str] = None,
):
    """Convenience: load both train+test in the paper's preprocessing.

    Returns:
        X_train, y_train, len_train, X_test, y_test, len_test
    """
    loader = SHDBinnedLoader(
        data_path=data_path,
        bin_size_ms=bin_size_ms,
        collapse_factor=collapse_factor,
        max_duration_ms=max_duration_ms,
    )
    X_tr, y_tr, L_tr = loader.get_dataset_binned(
        "train",
        max_samples_per_class=train_samples_per_class,
        target_classes=target_classes,
        binarize=binarize,
        dtype=dtype,
    )
    X_te, y_te, L_te = loader.get_dataset_binned(
        "test",
        max_samples_per_class=test_samples_per_class,
        target_classes=target_classes,
        binarize=binarize,
        dtype=dtype,
    )
    return X_tr, y_tr, L_tr, X_te, y_te, L_te


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="SHD count-based preprocessing (sparch / Fabre et al. 2025 style)."
    )
    p.add_argument("--bin_size_ms", type=float, default=4.0,
                   help="Time bin width in ms (paper default 4.0; also try 10 or 14).")
    p.add_argument("--collapse_factor", type=int, default=5,
                   help="Spatial collapsing: sum-pool every N consecutive channels (paper default 5).")
    p.add_argument("--max_duration_ms", type=float, default=1400.0,
                   help="Fixed window length in ms; spikes past this are dropped (paper default 1400).")
    p.add_argument("--binarize", action="store_true",
                   help="Cap per-bin counts to {0,1} (default: keep counts).")
    p.add_argument("--train_per_class", type=int, default=None)
    p.add_argument("--test_per_class", type=int, default=None)
    args = p.parse_args()

    X_tr, y_tr, L_tr, X_te, y_te, L_te = load_shd_binned(
        bin_size_ms=args.bin_size_ms,
        collapse_factor=args.collapse_factor,
        max_duration_ms=args.max_duration_ms,
        binarize=args.binarize,
        train_samples_per_class=args.train_per_class,
        test_samples_per_class=args.test_per_class,
    )
    print(f"\nTrain X: {X_tr.shape}  y: {y_tr.shape}  lengths in [{L_tr.min()}, {L_tr.max()}]")
    print(f"Test  X: {X_te.shape}  y: {y_te.shape}  lengths in [{L_te.min()}, {L_te.max()}]")
