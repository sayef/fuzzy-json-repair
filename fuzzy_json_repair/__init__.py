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

__version__ = "0.1.5"

# Main API - clean and simple
from .repair import ErrorType, RepairError, RepairFailedError, RepairResult, repair_keys

# Build __all__ dynamically based on optional dependencies
__all__ = [
    "repair_keys",
    "RepairError",
    "RepairResult",
    "RepairFailedError",
    "ErrorType",
    "__version__",
]

# Add Pydantic-dependent functions only if Pydantic is available
try:
    import pydantic  # noqa: F401

    __all__.append("fuzzy_model_validate_json")
except ImportError:
    pass


def __getattr__(name: str) -> Any:
    if name == "fuzzy_model_validate_json":
        # Check for Pydantic availability at import time
        try:
            import pydantic  # noqa: F401
        except ImportError as exc:
            msg = (
                "fuzzy_model_validate_json requires Pydantic. "
                "Install it with: pip install fuzzy-json-repair[pydantic]"
            )
            raise ImportError(msg) from exc

        from .repair import fuzzy_model_validate_json

        return fuzzy_model_validate_json
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
