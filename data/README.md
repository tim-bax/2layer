# Shared data loading

This package provides SHD and NMNIST data loading for **1layer_version** and **2layer_version** run scripts. It replaces the previous dependency on the `1layer/` and `2layer/` folders for data.

- **SHD**: `load_shd_data`, `create_shd_input_jax`, `SHDDataLoader` (from `data/shd.py`)
- **NMNIST**: `load_nmnist_data`, `create_nmnist_input_jax`, `NMNISTDataLoader` (from `data/nmnist.py`)

Usage from project root:

```python
import sys
sys.path.insert(0, "/path/to/1:2layer")  # or run with cwd = project root
from data import load_shd_data, create_shd_input_jax
# or
from data import load_nmnist_data, create_nmnist_input_jax
```

After this refactor, you can delete the **1layer** and **2layer** folders if you only use **1layer_version** and **2layer_version**.
