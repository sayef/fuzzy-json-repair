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

__version__ = "0.1.2"

# Main API - clean and simple
from .repair import (
    ErrorType,
    RepairError,
    RepairResult,
    fuzzy_model_validate_json,
    repair_keys,
)

__all__ = [
    "repair_keys",
    "fuzzy_model_validate_json",
    "RepairError",
    "RepairResult",
    "ErrorType",
    "__version__",
]
