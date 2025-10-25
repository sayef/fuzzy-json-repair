"""
Simple fuzzy JSON key repair using RapidFuzz.

Clean API:
- repair_keys(data, json_schema) - Repair dict keys using JSON schema
- Uses fuzz.ratio for fuzzy matching (no raw Levenshtein)
- Batch processing with cdist (when numpy available)
"""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from rapidfuzz import fuzz, process

try:
    import numpy as np  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class ErrorType(StrEnum):
    """Types of repair errors."""

    misspelled_key = "misspelled_key"
    unrecognized_key = "unrecognized_key"
    missing_expected_key = "missing_expected_key"


@dataclass
class RepairError:
    """Represents a single repair error."""

    error_type: ErrorType
    from_key: str | None = None
    to_key: str | None = None
    error_ratio: float = 0.0
    message: str | None = None

    def __str__(self) -> str:
        if self.error_type == ErrorType.misspelled_key:
            return (
                f"Misspelled key '{self.from_key}' → '{self.to_key}' "
                f"(error: {self.error_ratio:.1%})"
            )
        elif self.error_type == ErrorType.unrecognized_key:
            return f"Unrecognized key '{self.from_key}'"
        elif self.error_type == ErrorType.missing_expected_key:
            return f"Missing expected key '{self.to_key}'"
        else:
            return f"{self.error_type.value}: {self.message}"


@dataclass
class RepairResult:
    """Result of repair operation."""

    success: bool
    repaired_data: dict[str, Any] | None = None
    errors: list[RepairError] = field(default_factory=list)
    total_error_ratio: float = 0.0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def misspelled_keys(self) -> list[RepairError]:
        return [e for e in self.errors if e.error_type == ErrorType.misspelled_key]

    @property
    def missing_keys(self) -> list[RepairError]:
        return [e for e in self.errors if e.error_type == ErrorType.missing_expected_key]

    @property
    def unrecognized_keys(self) -> list[RepairError]:
        return [e for e in self.errors if e.error_type == ErrorType.unrecognized_key]


def _batch_match_with_cdist(
    input_keys: list[str], expected_keys: list[str], max_error_ratio: float
) -> dict[str, tuple[str | None, float]]:
    """Batch matching using cdist (requires numpy)."""
    similarity_matrix = process.cdist(
        input_keys,
        expected_keys,
        scorer=fuzz.ratio,
        score_cutoff=(1.0 - max_error_ratio) * 100,
        workers=1,
    )

    results: dict[str, tuple[str | None, float]] = {}
    used_expected_indices = set()

    for i, input_key in enumerate(input_keys):
        best_match = None
        best_similarity = -1
        best_idx = -1

        for j, expected_key in enumerate(expected_keys):
            if j in used_expected_indices:
                continue

            similarity = similarity_matrix[i][j]
            min_similarity = (1.0 - max_error_ratio) * 100

            if similarity < min_similarity:
                continue

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = expected_key
                best_idx = j

        if best_match:
            error_ratio = 1.0 - (best_similarity / 100.0)
            results[input_key] = (best_match, error_ratio)
            used_expected_indices.add(best_idx)
        else:
            results[input_key] = (None, 1.0)

    return results


def _batch_match_with_extract(
    input_keys: list[str], expected_keys: list[str], max_error_ratio: float
) -> dict[str, tuple[str | None, float]]:
    """Batch matching using extract loop (no numpy required)."""
    results: dict[str, tuple[str | None, float]] = {}
    available_expected = list(expected_keys)

    for input_key in input_keys:
        if not available_expected:
            results[input_key] = (None, 1.0)
            continue

        match_result = process.extractOne(
            input_key,
            available_expected,
            scorer=fuzz.ratio,
            score_cutoff=(1.0 - max_error_ratio) * 100,
        )

        if match_result:
            best_match, similarity, _ = match_result
            error_ratio = 1.0 - (similarity / 100.0)
            results[input_key] = (best_match, error_ratio)
            available_expected.remove(best_match)
        else:
            results[input_key] = (None, 1.0)

    return results


def _find_best_matches_batch(
    input_keys: list[str], expected_keys: list[str], max_error_ratio: float = 0.3
) -> dict[str, tuple[str | None, float]]:
    """Find best matches for multiple keys using batch processing."""
    if not input_keys or not expected_keys:
        return dict.fromkeys(input_keys, (None, 1.0))

    if HAS_NUMPY:
        return _batch_match_with_cdist(input_keys, expected_keys, max_error_ratio)
    else:
        return _batch_match_with_extract(input_keys, expected_keys, max_error_ratio)


def _extract_schema_info(
    json_schema: dict[str, Any],
) -> tuple[set[str], set[str], dict[str, Any], dict[str, Any]]:
    """
    Extract schema information from Pydantic JSON schema.

    Returns:
        (expected_keys, required_keys, nested_schemas, list_item_schemas)
    """
    properties = json_schema.get("properties", {})
    required = set(json_schema.get("required", []))
    expected_keys = set(properties.keys())
    nested_schemas = {}
    list_item_schemas = {}

    # Check for $defs (Pydantic puts nested models here)
    defs = json_schema.get("$defs", {})

    for key, prop_schema in properties.items():
        # Check if it's a nested object reference
        if "$ref" in prop_schema:
            # Resolve reference like "#/$defs/Address"
            ref = prop_schema["$ref"]
            if ref.startswith("#/$defs/"):
                def_name = ref.split("/")[-1]
                if def_name in defs:
                    resolved_schema = defs[def_name].copy()
                    # Include $defs in nested schema for further resolution
                    if "$defs" not in resolved_schema and defs:
                        resolved_schema["$defs"] = defs
                    nested_schemas[key] = resolved_schema

        # Direct nested object (no $ref)
        elif prop_schema.get("type") == "object" and "properties" in prop_schema:
            schema_copy = prop_schema.copy()
            # Include $defs for nested resolution
            if "$defs" not in schema_copy and defs:
                schema_copy["$defs"] = defs
            nested_schemas[key] = schema_copy

        # Array of objects
        elif prop_schema.get("type") == "array" and "items" in prop_schema:
            items = prop_schema["items"]

            # Reference to another model
            if "$ref" in items:
                ref = items["$ref"]
                if ref.startswith("#/$defs/"):
                    def_name = ref.split("/")[-1]
                    if def_name in defs:
                        resolved_schema = defs[def_name].copy()
                        # Include $defs in list item schema
                        if "$defs" not in resolved_schema and defs:
                            resolved_schema["$defs"] = defs
                        list_item_schemas[key] = resolved_schema

            # Inline object definition
            elif items.get("type") == "object" and "properties" in items:
                schema_copy = items.copy()
                # Include $defs for nested resolution
                if "$defs" not in schema_copy and defs:
                    schema_copy["$defs"] = defs
                list_item_schemas[key] = schema_copy

    return expected_keys, required, nested_schemas, list_item_schemas


def repair_keys(
    data: dict[str, Any],
    json_schema: dict[str, Any],
    max_error_ratio_per_key: float = 0.3,
    max_total_error_ratio: float = 0.5,
    strict_validation: bool = False,
) -> tuple[dict[str, Any] | None, float, list[RepairError]]:
    """
    Repair dictionary keys using JSON schema from Pydantic.

    Args:
        data: Input dictionary with potential typos
        json_schema: JSON schema dict (from model.model_json_schema())
        max_error_ratio_per_key: Maximum error ratio per individual key (0.0-1.0)
        max_total_error_ratio: Maximum average error ratio across all schema fields (0.0-1.0)
        strict_validation: If True, reject unrecognized keys

    Returns:
        (repaired_data, total_error_ratio, errors)
        - repaired_data is None if repair exceeds acceptable thresholds
        - total_error_ratio and errors are always returned for diagnostics

    Example:
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        schema = User.model_json_schema()
        data = {'nam': 'John', 'agge': 30}

        repaired, ratio, errors = repair_keys(data, schema)
        if repaired is None:
            print("Repair failed - too many errors")
        else:
            user = User.model_validate(repaired)
    """
    repaired_data: dict[str, Any] = {}
    total_error_ratio: float = 0.0
    errors: list[RepairError] = []

    # Extract schema information
    expected_keys, required_keys, nested_schemas, list_item_schemas = _extract_schema_info(
        json_schema
    )
    matched_expected_keys: set[str] = set()

    # Separate exact matches from fuzzy candidates
    exact_matches = {k: v for k, v in data.items() if k in expected_keys}
    fuzzy_candidates = {k: v for k, v in data.items() if k not in expected_keys}

    # Process exact matches (fast path)
    for key, value in exact_matches.items():
        matched_expected_keys.add(key)

        # Handle nested objects
        if key in nested_schemas and isinstance(value, dict):
            nested_data, nested_ratio, nested_errors = repair_keys(
                value,
                nested_schemas[key],
                max_error_ratio_per_key,
                max_total_error_ratio,
                strict_validation,
            )
            # Propagate nested repair failure
            if nested_data is None:
                return None, total_error_ratio + nested_ratio, errors + nested_errors
            repaired_data[key] = nested_data
            total_error_ratio += nested_ratio
            errors.extend(nested_errors)

        # Handle lists of objects
        elif key in list_item_schemas and isinstance(value, list):
            repaired_list = []
            for item in value:
                if isinstance(item, dict):
                    item_data, item_ratio, item_errors = repair_keys(
                        item,
                        list_item_schemas[key],
                        max_error_ratio_per_key,
                        max_total_error_ratio,
                        strict_validation,
                    )
                    # Propagate list item repair failure
                    if item_data is None:
                        return None, total_error_ratio + item_ratio, errors + item_errors
                    repaired_list.append(item_data)
                    total_error_ratio += item_ratio
                    errors.extend(item_errors)
                else:
                    repaired_list.append(item)
            repaired_data[key] = repaired_list

        # Primitive value
        else:
            repaired_data[key] = value

    # Batch fuzzy matching for remaining keys
    if fuzzy_candidates:
        unmatched_expected = expected_keys - matched_expected_keys
        fuzzy_keys = list(fuzzy_candidates.keys())

        matches = _find_best_matches_batch(
            fuzzy_keys, list(unmatched_expected), max_error_ratio_per_key
        )

        for input_key, (matched_key, error_ratio) in matches.items():
            value = fuzzy_candidates[input_key]

            if matched_key and error_ratio <= max_error_ratio_per_key:
                matched_expected_keys.add(matched_key)

                errors.append(
                    RepairError(
                        error_type=ErrorType.misspelled_key,
                        from_key=input_key,
                        to_key=matched_key,
                        error_ratio=error_ratio,
                    )
                )
                total_error_ratio += error_ratio

                # Handle nested objects for repaired keys
                if matched_key in nested_schemas and isinstance(value, dict):
                    nested_data, nested_ratio, nested_errors = repair_keys(
                        value,
                        nested_schemas[matched_key],
                        max_error_ratio_per_key,
                        max_total_error_ratio,
                        strict_validation,
                    )
                    # Propagate nested repair failure
                    if nested_data is None:
                        return None, total_error_ratio + nested_ratio, errors + nested_errors
                    repaired_data[matched_key] = nested_data
                    total_error_ratio += nested_ratio
                    errors.extend(nested_errors)

                # Handle lists of objects for repaired keys
                elif matched_key in list_item_schemas and isinstance(value, list):
                    repaired_list = []
                    for item in value:
                        if isinstance(item, dict):
                            item_data, item_ratio, item_errors = repair_keys(
                                item,
                                list_item_schemas[matched_key],
                                max_error_ratio_per_key,
                                max_total_error_ratio,
                                strict_validation,
                            )
                            # Propagate list item repair failure
                            if item_data is None:
                                return None, total_error_ratio + item_ratio, errors + item_errors
                            repaired_list.append(item_data)
                            total_error_ratio += item_ratio
                            errors.extend(item_errors)
                        else:
                            repaired_list.append(item)
                    repaired_data[matched_key] = repaired_list

                # Primitive value
                else:
                    repaired_data[matched_key] = value
            else:
                # No good match
                if not strict_validation:
                    repaired_data[input_key] = value

                errors.append(
                    RepairError(
                        error_type=ErrorType.unrecognized_key,
                        from_key=input_key,
                        error_ratio=1.0,
                        message=f"Key '{input_key}' not recognized and no close match found",
                    )
                )

    # Check for missing required fields
    missing_keys = required_keys - matched_expected_keys
    for missing_key in missing_keys:
        errors.append(
            RepairError(
                error_type=ErrorType.missing_expected_key,
                to_key=missing_key,
                error_ratio=1.0,
                message=f"Required key '{missing_key}' is missing from input",
            )
        )

    # Validate acceptability of repairs
    num_fields = len(json_schema.get("properties", {}))

    if num_fields > 0:
        # Calculate average error ratio across schema fields
        avg_error_ratio = total_error_ratio / num_fields

        # Check individual error ratios
        misspelled = [e for e in errors if e.error_type == ErrorType.misspelled_key]
        if misspelled:
            max_individual_error = max(e.error_ratio for e in misspelled)
            # Reject if any single repair has very high error
            if max_individual_error > min(0.5, max_error_ratio_per_key * 1.5):
                return None, total_error_ratio, errors

        # Reject if average error is too high
        if avg_error_ratio > max_total_error_ratio:
            return None, total_error_ratio, errors

    # In strict mode, reject unrecognized keys
    if strict_validation:
        has_unrecognized = any(e.error_type == ErrorType.unrecognized_key for e in errors)
        if has_unrecognized:
            return None, total_error_ratio, errors

    # Check for fundamental mismatch: many errors, no successful repairs
    missing_required_errors = [e for e in errors if e.error_type == ErrorType.missing_expected_key]
    unrecognized_errors = [e for e in errors if e.error_type == ErrorType.unrecognized_key]
    successful_repairs = [e for e in errors if e.error_type == ErrorType.misspelled_key]

    if (
        len(unrecognized_errors) > 0
        and len(successful_repairs) == 0
        and len(missing_required_errors) > 3
    ):
        return None, total_error_ratio, errors

    return repaired_data, total_error_ratio, errors


def fuzzy_model_validate_json(
    json_data: str,
    model_cls: type[BaseModel],
    repair_syntax: bool = True,
    max_error_ratio_per_key: float = 0.3,
    max_total_error_ratio: float = 0.3,
    strict_validation: bool = False,
) -> Any:
    """
    Repair LLM response JSON data to match a Pydantic model schema.

    This is a convenience function that:
    1. Optionally repairs JSON syntax errors
    2. Tries fast path validation
    3. Falls back to fuzzy key repair if needed
    4. Validates acceptability of repairs
    5. Returns validated Pydantic instance

    Args:
        json_data: JSON string to repair and validate
        model_cls: Pydantic model class to match against
        repair_syntax: If True, attempt to repair JSON syntax first (requires json-repair-py)
        max_error_ratio_per_key: Maximum allowed error ratio per individual key (0.0-1.0)
        max_total_error_ratio: Maximum average error ratio across all fields (0.0-1.0)
        strict_validation: If True, fail on any unrecognized keys

    Returns:
        Validated Pydantic BaseModel instance

    Raises:
        ValueError: If repair fails or validation fails after repair

    Example:
        from pydantic import BaseModel
        from fuzzy_json_repair import fuzzy_model_validate_json

        class User(BaseModel):
            name: str
            age: int

        json_str = '{"nam": "John", "agge": 30}'
        user = fuzzy_model_validate_json(json_str, User)
        # Returns: User(name='John', age=30)
    """
    # Attempt to repair JSON syntax if repair_syntax is True
    if repair_syntax:
        try:
            import json_repair

            repaired_json = json_repair.repair_json(json_data, skip_json_loads=True)
        except ImportError:
            repaired_json = json_data
    else:
        repaired_json = json_data

    # Try fast path first with strict validation to catch unknown fields
    try:
        # Use strict=True to detect unknown fields that might be typos
        return model_cls.model_validate_json(repaired_json, strict=True)
    except Exception:
        # Strict validation failed - likely due to unknown fields or other issues
        # Fall back to key repair to handle typos and preserve data
        pass

    # Parse JSON
    try:
        parsed_data = json.loads(repaired_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON syntax error: {e}") from e

    if not isinstance(parsed_data, dict):
        raise ValueError(f"Expected dict, got {type(parsed_data)}")

    # Get schema and repair keys
    schema = model_cls.model_json_schema()
    repaired_data, total_error_ratio, errors = repair_keys(
        parsed_data, schema, max_error_ratio_per_key, max_total_error_ratio, strict_validation
    )

    # Check if repair succeeded
    if repaired_data is not None:
        try:
            # Validate the repaired data
            return model_cls.model_validate(repaired_data)
        except Exception as validation_error:
            # Repair succeeded but validation failed
            raise ValueError(
                f"Validation failed after key repair for {model_cls.__name__}: "
                f"{str(validation_error)}"
            ) from validation_error

    # Repair failed - create detailed error message
    error_summary = []
    misspelled_keys = [e for e in errors if e.error_type == ErrorType.misspelled_key]
    missing_keys = [e for e in errors if e.error_type == ErrorType.missing_expected_key]
    unrecognized_keys = [e for e in errors if e.error_type == ErrorType.unrecognized_key]

    if misspelled_keys:
        error_summary.append(f"{len(misspelled_keys)} misspelled keys")
    if missing_keys:
        error_summary.append(f"{len(missing_keys)} missing keys")
    if unrecognized_keys:
        error_summary.append(f"{len(unrecognized_keys)} unrecognized keys")

    raise ValueError(
        f"JSON repair failed for {model_cls.__name__}. "
        f"Issues found: {', '.join(error_summary)}. "
        f"Details: {[str(e) for e in errors[:5]]}"  # Show first 5 errors
    )
