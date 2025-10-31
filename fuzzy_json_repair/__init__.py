"""
Fuzzy JSON Repair - Fix typos in JSON keys using fuzzy matching.

Simple API:
    from fuzzy_json_repair import repair_keys, fuzzy_model_validate_json

    # Low-level: Repair dict keys using JSON schema
    schema = MyModel.model_json_schema()
    result = repair_keys(data, schema)
    if result.success:
        use(result.data)

    # High-level: Repair JSON string and return validated Pydantic model
    user = fuzzy_model_validate_json(json_str, User)

Uses:
- fuzz.ratio from RapidFuzz (no raw Levenshtein)
- Batch processing with cdist (when numpy available)
- Direct JSON schema from Pydantic
"""

from typing import Any

__version__ = "0.1.4"

# Main API - clean and simple
from .repair import ErrorType, RepairError, RepairFailedError, RepairResult, repair_keys

__all__ = [
    "repair_keys",
    "fuzzy_model_validate_json",
    "RepairError",
    "RepairResult",
    "RepairFailedError",
    "ErrorType",
    "__version__",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    if name == "fuzzy_model_validate_json":
        from .repair import fuzzy_model_validate_json

        return fuzzy_model_validate_json
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
