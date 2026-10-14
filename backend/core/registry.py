"""
backend/core/registry.py  --  SEAM 4
====================================

Owner: W4. One dict, and the accessors around it.

Adding a Phase-2 model means adding one line here and zero lines in
backend/ or frontend/. That is the whole point of the seam: W2 renders the
model picker from GET /models/catalog, so a model that exists here appears in
the UI on the next page load with its parameter controls already generated from
`param_schema`.

If you ever find yourself writing `if model_name == "qsvc"` outside this file,
the seam has leaked.
"""

from __future__ import annotations

from .contracts import BaseModel, Kind
from .models_classical import RandomForestModel, SVCModel
from .models_quantum import QSVCModel, VQCModel

__all__ = ["MODEL_REGISTRY", "get_model", "catalog", "kind_of", "DEFAULT_MODELS"]


MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    SVCModel.name: SVCModel,
    RandomForestModel.name: RandomForestModel,
    QSVCModel.name: QSVCModel,
    VQCModel.name: VQCModel,
}

# What the Configure screen ticks on first load. QSVC rather than VQC because
# it is the fast one and the one the research claim rests on; VQC is opt-in.
DEFAULT_MODELS: list[str] = ["svc", "rf", "qsvc"]


def get_model(name: str) -> type[BaseModel]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}"
        ) from None


def kind_of(name: str) -> Kind:
    return get_model(name).kind


def catalog() -> list[dict]:
    """GET /models/catalog. Classical first, then quantum, so the two columns
    in the model picker come out in a stable order without W2 sorting them."""
    entries = [m.catalog_entry() for m in MODEL_REGISTRY.values()]
    return sorted(entries, key=lambda e: (e["kind"] != "classical", e["name"]))


# ---------------------------------------------------------------------------
# Structural checks:  python -m backend.core.registry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for key, cls in MODEL_REGISTRY.items():
        assert cls.name == key, f"{cls.__name__}.name != its registry key"
        assert cls.label, f"{key}: no label -- the UI would render a blank row"
        assert cls.kind in ("classical", "quantum"), key
        # param_schema must be renderable by a generic ParamControl.
        for pname, schema in cls.param_schema.items():
            assert "type" in schema and "default" in schema, f"{key}.{pname}"
            assert schema["type"] in ("int", "float", "bool", "enum"), f"{key}.{pname}"

    names = [e["name"] for e in catalog()]
    assert names == ["rf", "svc", "qsvc", "vqc"], names
    assert set(DEFAULT_MODELS) <= set(MODEL_REGISTRY)

    print(f"{len(MODEL_REGISTRY)} models registered")
    for e in catalog():
        knobs = ", ".join(e["param_schema"]) or "-"
        print(f"  {e['kind']:9s} {e['name']:6s} {e['label']:34s} [{knobs}]")
    print("registry OK")
