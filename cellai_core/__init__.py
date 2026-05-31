"""
cellai_core
===========
Shared mathematical core for Cell AI v1 and v2.

Public API
----------
ModelParams       — unified configuration dataclass
CellularPDE       — vectorised N-partition PDE step (GPU-native)
PartitionManager  — manages CellularPDE state for a model
UniversalEncoder  — BPE text → cellular state vector
MemoryFormation   — temporal memory kernel integration
MetaplasticityLayer — sliding-threshold plasticity gate
set_seed          — reproducibility helper
"""
from cellai_core.base    import ModelParams, CellularPDE, ring_neighbors
from cellai_core.encoder import UniversalEncoder
from cellai_core.memory  import MemoryFormation, MetaplasticityLayer, memory_kernel, weight_kernel
from cellai_core.partition import PartitionManager
from cellai_core.utils   import set_seed, run_benchmark, CProfileContext

__all__ = [
    "ModelParams", "CellularPDE", "ring_neighbors",
    "UniversalEncoder",
    "MemoryFormation", "MetaplasticityLayer", "memory_kernel", "weight_kernel",
    "PartitionManager",
    "set_seed", "run_benchmark", "CProfileContext",
]
