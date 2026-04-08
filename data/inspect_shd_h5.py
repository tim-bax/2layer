#!/usr/bin/env python3
"""
Inspect SHD HDF5 content and summarize the first N samples.

Examples:
  python data/inspect_shd_h5.py --split train --n_samples 5
  python data/inspect_shd_h5.py --file "/Users/tbax/Documents/Heidelberg_Data/hdspikes/shd_train.h5"
"""

import argparse
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


USE_H5PY = False
try:
    import tables  # type: ignore
except (ImportError, ValueError):
    USE_H5PY = True
try:
    import h5py  # type: ignore
except ImportError:
    if USE_H5PY:
        raise ImportError("Need either 'tables' or 'h5py' installed to inspect SHD HDF5.")
    h5py = None


def _default_data_dir() -> str:
    if "SHD_DATA_PATH" in os.environ:
        return os.environ["SHD_DATA_PATH"]
    return os.path.expanduser("~/Documents/Heidelberg_Data/hdspikes")


def _parse_args():
    p = argparse.ArgumentParser(description="Inspect SHD HDF5 schema and first samples")
    p.add_argument("--split", choices=["train", "test"], default="train", help="SHD split file to inspect")
    p.add_argument("--data_dir", type=str, default=None, help="Directory containing shd_train.h5 / shd_test.h5")
    p.add_argument("--file", type=str, default=None, help="Explicit HDF5 file path (overrides --split/--data_dir)")
    p.add_argument("--n_samples", type=int, default=5, help="Number of samples to summarize")
    p.add_argument("--events_head", type=int, default=12, help="How many (unit,time) events to print per sample")
    return p.parse_args()


def _safe_attr_dict_h5py(obj) -> Dict[str, Any]:
    out = {}
    for k in obj.attrs.keys():
        try:
            v = obj.attrs[k]
            if hasattr(v, "tolist"):
                v = v.tolist()
            out[k] = v
        except Exception:
            out[k] = "<unreadable>"
    return out


def _print_h5py_tree(name: str, obj):
    kind = "Group" if isinstance(obj, h5py.Group) else "Dataset"
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    print(f"  - {name} [{kind}] shape={shape} dtype={dtype}")


def _print_tables_tree(node, indent="  "):
    print(f"{indent}- {node._v_pathname} [{node.__class__.__name__}]")
    if hasattr(node, "dtype"):
        print(f"{indent}  dtype={getattr(node, 'dtype', None)} shape={getattr(node, 'shape', None)}")
    if hasattr(node, "_v_attrs"):
        attr_keys = list(node._v_attrs._f_list("user"))
        if attr_keys:
            print(f"{indent}  attrs={attr_keys}")


def _open_and_inspect_h5py(path: str, n_samples: int, events_head: int):
    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        print(f"Backend: h5py")
        print("\nTop-level attrs:")
        attrs = _safe_attr_dict_h5py(f)
        if attrs:
            for k, v in attrs.items():
                print(f"  {k}: {v}")
        else:
            print("  (none)")

        print("\nTree:")
        f.visititems(_print_h5py_tree)

        labels = f["labels"]
        units = f["spikes"]["units"]
        times = f["spikes"]["times"]
        n_total = int(labels.shape[0])
        print("\nCore datasets:")
        print(f"  labels: shape={labels.shape}, dtype={labels.dtype}")
        print(f"  spikes/units: shape={units.shape}, dtype={units.dtype}")
        print(f"  spikes/times: shape={times.shape}, dtype={times.dtype}")
        print(f"  total samples: {n_total}")

        extra_fields = [k for k in f.keys() if k not in ("spikes", "labels")]
        if extra_fields:
            print(f"\nExtra top-level fields: {extra_fields}")

        n_show = max(0, min(n_samples, n_total))
        print(f"\nFirst {n_show} samples:")
        for i in range(n_show):
            lbl = int(labels[i])
            u = np.asarray(units[i])
            t = np.asarray(times[i])
            n_events = int(len(u))
            n_active = int(np.unique(u).size) if n_events > 0 else 0
            t_ms = t * 1000.0
            head_n = min(events_head, n_events)
            event_head = [(int(u[j]), float(t_ms[j])) for j in range(head_n)]
            print(f"\n  sample[{i}]")
            print(f"    label: {lbl}")
            print(f"    n_events: {n_events}")
            print(f"    n_active_units: {n_active}")
            if n_events > 0:
                print(f"    unit_range: [{int(u.min())}, {int(u.max())}]")
                print(f"    time_range_sec: [{float(t.min()):.6f}, {float(t.max()):.6f}]")
                print(f"    time_range_ms: [{float(t_ms.min()):.3f}, {float(t_ms.max()):.3f}]")
                print(f"    first_{head_n}_events_(unit,time_ms): {event_head}")


def _open_and_inspect_tables(path: str, n_samples: int, events_head: int):
    f = tables.open_file(path, "r")
    try:
        print(f"File: {path}")
        print(f"Backend: PyTables")
        print("\nTop-level attrs:")
        root_attrs = list(f.root._v_attrs._f_list("user"))
        if root_attrs:
            for k in root_attrs:
                print(f"  {k}: {getattr(f.root._v_attrs, k)}")
        else:
            print("  (none)")

        print("\nTree:")
        for node in f.walk_nodes("/", classname=None):
            _print_tables_tree(node)

        labels = f.root.labels
        units = f.root.spikes.units
        times = f.root.spikes.times
        n_total = int(len(labels))
        print("\nCore datasets:")
        print(f"  labels: len={len(labels)}, dtype={labels.dtype}")
        print(f"  spikes/units: len={len(units)}, dtype={units.dtype}")
        print(f"  spikes/times: len={len(times)}, dtype={times.dtype}")
        print(f"  total samples: {n_total}")

        root_children = [n._v_name for n in f.root._f_list_nodes()]
        extra_fields = [k for k in root_children if k not in ("spikes", "labels")]
        if extra_fields:
            print(f"\nExtra top-level fields: {extra_fields}")

        n_show = max(0, min(n_samples, n_total))
        print(f"\nFirst {n_show} samples:")
        for i in range(n_show):
            lbl = int(labels[i])
            u = np.asarray(units[i])
            t = np.asarray(times[i])
            n_events = int(len(u))
            n_active = int(np.unique(u).size) if n_events > 0 else 0
            t_ms = t * 1000.0
            head_n = min(events_head, n_events)
            event_head = [(int(u[j]), float(t_ms[j])) for j in range(head_n)]
            print(f"\n  sample[{i}]")
            print(f"    label: {lbl}")
            print(f"    n_events: {n_events}")
            print(f"    n_active_units: {n_active}")
            if n_events > 0:
                print(f"    unit_range: [{int(u.min())}, {int(u.max())}]")
                print(f"    time_range_sec: [{float(t.min()):.6f}, {float(t.max()):.6f}]")
                print(f"    time_range_ms: [{float(t_ms.min()):.3f}, {float(t_ms.max()):.3f}]")
                print(f"    first_{head_n}_events_(unit,time_ms): {event_head}")
    finally:
        f.close()


def main():
    args = _parse_args()
    if args.file:
        h5_path = args.file
    else:
        base = args.data_dir if args.data_dir is not None else _default_data_dir()
        h5_path = os.path.join(base, f"shd_{args.split}.h5")

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    if USE_H5PY:
        _open_and_inspect_h5py(h5_path, args.n_samples, args.events_head)
    else:
        _open_and_inspect_tables(h5_path, args.n_samples, args.events_head)


if __name__ == "__main__":
    main()
