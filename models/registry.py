"""
models.registry
===============
Factory for all (model_type, version) combinations.

Supported versions:  v1, v2
Supported models:    nlp, nlp_trad, math, math_trad, software, software_trad, cot, multimodal

Usage:
    from models.registry import get_model
    m = get_model("nlp", version="v2")
    print(m.chat("Explain attention mechanisms."))
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from cellai_core.base import ModelParams


def _get_cell_class(version: str):
    if version == "v1":
        from v1.cell_ai import CellAI
        return CellAI
    if version == "v2":
        from v2.cell_ai_v2 import CellAIv2
        return CellAIv2
    raise ValueError(f"Unknown version {version!r}. Choose from: {list_versions()}")


def _get_model_class(model_type: str):
    dispatch = {
        "nlp":           ("models.nlp.new_nlp",             "NewNLPModel"),
        "nlp_trad":      ("models.nlp.trad_nlp",            "TradNLPModel"),
        "math":          ("models.math.new_math",            "NewMathModel"),
        "math_trad":     ("models.math.trad_math",           "TradMathModel"),
        "software":      ("models.software.new_code",        "NewSoftwareModel"),
        "software_trad": ("models.software.trad_code",       "TradSoftwareModel"),
        "cot":           ("models.thinking_cot.thinking_cot","ThinkingCoTModel"),
        "multimodal":    ("models.multimodal.multimodal",    "MultiModalModel"),
    }
    if model_type not in dispatch:
        raise ValueError(f"Unknown model {model_type!r}. Choose from: {list_models()}")
    module_path, class_name = dispatch[model_type]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def get_model(
    model_type: str,
    version: str = "v1",
    params: Optional[ModelParams] = None,
    **cell_kwargs: Any,
) -> Any:
    """
    Instantiate a domain model with the given Cell AI version as backbone.

    Args:
        model_type:   one of nlp, nlp_trad, math, math_trad,
                      software, software_trad, cot, multimodal
        version:      v1 or v2
        params:       ModelParams (auto-created if not given)
        **cell_kwargs: forwarded to the CellAI constructor

    Returns:
        A CellAIModel subclass instance.
    """
    params       = params or ModelParams()
    CellClass    = _get_cell_class(version)
    ModelClass   = _get_model_class(model_type)
    cell_system  = CellClass(params=params, **cell_kwargs)
    return ModelClass(cell_system=cell_system, params=params)


def list_models() -> List[str]:
    return ["nlp", "nlp_trad", "math", "math_trad",
            "software", "software_trad", "cot", "multimodal"]


def list_versions() -> List[str]:
    return ["v1", "v2"]
