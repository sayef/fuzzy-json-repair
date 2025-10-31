"""
Simple fuzzy JSON key repair using RapidFuzz.

Clean API:
- repair_keys(data, json_schema) - Repair dict keys using JSON schema
- Uses fuzz.ratio for fuzzy matching (no raw Levenshtein)
- Batch processing with cdist (when numpy available)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - only used for type hints
    from pydantic import BaseModel
import json_repair
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


class RepairFailedError(ValueError):
    """Exception raised when JSON repair fails.

    Provides structured access to repair attempt details including all errors,
    paths, and categorization.

    Attributes:
        result: The RepairResult object containing all error details
        model_cls: The Pydantic model class that was being validated against
        json_data: The original JSON string that failed to repair
    """

    def __init__(
        self,
        message: str,
        result: RepairResult,
        model_cls: type[BaseModel] | None = None,
        json_data: str | None = None,
    ):
        super().__init__(message)
        self.result = result
        self.model_cls = model_cls
        self.json_data = json_data

    @property
    def errors(self) -> list[RepairError]:
        """All errors encountered during repair."""
        return self.result.errors

    @property
    def repaired_errors(self) -> list[RepairError]:
        """Errors that were successfully handled (repaired or dropped)."""
        return self.result.repaired_errors

    @property
    def unrepaired_errors(self) -> list[RepairError]:
        """Errors that could not be repaired."""
        return self.result.unrepaired_errors

    @property
    def error_ratio(self) -> float:
        """Total error ratio from repair attempt."""
        return self.result.error_ratio


@dataclass
class RepairError:
    """Represents a single repair error."""

    error_type: ErrorType
    from_key: str | None = None
    to_key: str | None = None
    error_ratio: float = 0.0
    message: str | None = None
    path: str | None = None  # JSON path where error occurred (e.g., "profile.address")

    def __str__(self) -> str:
        path_prefix = f"[{self.path}] " if self.path else ""
        if self.error_type == ErrorType.misspelled_key:
            return (
                f"{path_prefix}Misspelled key '{self.from_key}' → '{self.to_key}' "
                f"(error: {self.error_ratio:.1%})"
            )
        elif self.error_type == ErrorType.unrecognized_key:
            return f"{path_prefix}Unrecognized key '{self.from_key}'"
        elif self.error_type == ErrorType.missing_expected_key:
            return f"{path_prefix}Missing expected key '{self.to_key}'"
        else:
            return f"{path_prefix}{self.error_type.value}: {self.message}"


@dataclass
class RepairResult:
    """Result of a repair operation.

    Attributes:
        success: Whether the repair succeeded within acceptable thresholds
        data: Repaired data (None if repair failed)
        error_ratio: Total error ratio across all repairs
        errors: List of all errors encountered during repair
        repaired_errors: Subset of `errors` that were successfully repaired
        unrepaired_errors: Subset of `errors` that still requires attention
    """

    success: bool
    data: dict[str, Any] | None
    error_ratio: float
    errors: list[RepairError] = field(default_factory=list)
    repaired_errors: list[RepairError] = field(default_factory=list)
    unrepaired_errors: list[RepairError] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.repaired_errors and not self.unrepaired_errors and self.errors:
            for error in self.errors:
                if error.error_type == ErrorType.misspelled_key:
                    self.repaired_errors.append(error)
                else:
                    self.unrepaired_errors.append(error)

    @property
    def failed(self) -> bool:
        """True if repair failed."""
        return not self.success

    @property
    def has_errors(self) -> bool:
        """True if any errors were encountered."""
        return len(self.errors) > 0

    @property
    def misspelled_keys(self) -> list[RepairError]:
        """List of misspelled key errors."""
        return [e for e in self.errors if e.error_type == ErrorType.misspelled_key]

    @property
    def missing_keys(self) -> list[RepairError]:
        """List of missing key errors."""
        return [e for e in self.errors if e.error_type == ErrorType.missing_expected_key]

    @property
    def unrecognized_keys(self) -> list[RepairError]:
        """List of unrecognized key errors."""
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

    def resolve_ref(ref: str) -> dict[str, Any] | None:
        """Resolve a $ref to its schema definition."""
        if ref.startswith("#/$defs/"):
            def_name = ref.split("/")[-1]
            if def_name in defs:
                resolved_schema: dict[str, Any] = defs[def_name].copy()
                if "$defs" not in resolved_schema and defs:
                    resolved_schema["$defs"] = defs
                return resolved_schema
        return None

    def extract_from_schema(
        prop_schema: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Extract nested object schema and list item schema from a property schema.

        Returns: (nested_object_schema, list_item_schema)
        """
        # Direct $ref
        if "$ref" in prop_schema:
            return resolve_ref(prop_schema["$ref"]), None

        # Direct object
        if prop_schema.get("type") == "object" and "properties" in prop_schema:
            schema_copy = prop_schema.copy()
            if "$defs" not in schema_copy and defs:
                schema_copy["$defs"] = defs
            return schema_copy, None

        # Array of objects
        if prop_schema.get("type") == "array" and "items" in prop_schema:
            items = prop_schema["items"]
            if "$ref" in items:
                return None, resolve_ref(items["$ref"])
            if items.get("type") == "object" and "properties" in items:
                schema_copy = items.copy()
                if "$defs" not in schema_copy and defs:
                    schema_copy["$defs"] = defs
                return None, schema_copy

        # Union types (anyOf, oneOf, allOf) - extract first non-null object schema
        for union_key in ("anyOf", "oneOf", "allOf"):
            if union_key in prop_schema:
                for option in prop_schema[union_key]:
                    if option.get("type") == "null":
                        continue
                    obj_schema, list_schema = extract_from_schema(option)
                    if obj_schema or list_schema:
                        return obj_schema, list_schema

        return None, None

    for key, prop_schema in properties.items():
        obj_schema, list_schema = extract_from_schema(prop_schema)
        if obj_schema:
            nested_schemas[key] = obj_schema
        if list_schema:
            list_item_schemas[key] = list_schema

    return expected_keys, required, nested_schemas, list_item_schemas


def _check_min_items_constraint(
    repaired_list: list[Any],
    original_list: list[Any],
    key: str,
    json_schema: dict[str, Any],
) -> bool:
    """Check if dropping items would violate minItems constraint.

    Returns True if constraint is satisfied, False if violated.
    """
    if len(repaired_list) >= len(original_list):
        return True  # No items dropped

    properties = json_schema.get("properties", {})
    list_schema = properties.get(key, {})
    min_items: int = list_schema.get("minItems", 0)

    return len(repaired_list) >= min_items


def repair_keys(
    data: dict[str, Any],
    json_schema: dict[str, Any],
    max_error_ratio_per_key: float = 0.3,
    max_total_error_ratio: float = 0.5,
    strict_validation: bool = False,
    drop_unrepairable_items: bool = False,
) -> RepairResult:
    """
    Repair dictionary keys using JSON schema from Pydantic.

    Args:
        data: Input dictionary with potential typos
        json_schema: JSON schema dict (from model.model_json_schema())
        max_error_ratio_per_key: Maximum error ratio per individual key (0.0-1.0)
        max_total_error_ratio: Maximum average error ratio across all schema fields (0.0-1.0)
        strict_validation: If True, reject unrecognized keys
        drop_unrepairable_items: If True, drop list items, unrecognized keys, and optional
            nested objects that can't be repaired (respects minItems, preserves required fields)

    Returns:
        RepairResult with success flag, repaired data, error ratio, and error details

    Example:
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        schema = User.model_json_schema()
        data = {'nam': 'John', 'agge': 30}

        result = repair_keys(data, schema)
        if result.success:
            user = User.model_validate(result.data)
        else:
            print(f"Repair failed: {len(result.errors)} errors")
    """
    return _repair_keys(
        data,
        json_schema,
        max_error_ratio_per_key,
        max_total_error_ratio,
        strict_validation,
        drop_unrepairable_items,
        _path="",
    )


def _repair_keys(
    data: dict[str, Any],
    json_schema: dict[str, Any],
    max_error_ratio_per_key: float,
    max_total_error_ratio: float,
    strict_validation: bool,
    drop_unrepairable_items: bool,
    _path: str,
) -> RepairResult:
    """Internal implementation of repair_keys with path tracking."""
    repaired_data: dict[str, Any] = {}
    total_error_ratio: float = 0.0
    errors: list[RepairError] = []
    repaired_errors: list[RepairError] = []
    unrepaired_errors: list[RepairError] = []

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
            nested_path = f"{_path}.{key}" if _path else key
            nested_result = _repair_keys(
                value,
                nested_schemas[key],
                max_error_ratio_per_key,
                max_total_error_ratio,
                strict_validation,
                drop_unrepairable_items,
                nested_path,
            )
            if nested_result.success:
                repaired_data[key] = nested_result.data
                total_error_ratio += nested_result.error_ratio
                errors.extend(nested_result.errors)
                repaired_errors.extend(nested_result.repaired_errors)
                unrepaired_errors.extend(nested_result.unrepaired_errors)
            elif drop_unrepairable_items and key not in required_keys:
                # Drop optional nested object that failed repair, but record the errors
                matched_expected_keys.discard(key)
                errors.extend(nested_result.errors)
                repaired_errors.extend(nested_result.errors)  # Treated as "repaired" by dropping
                continue
            else:
                # Propagate nested repair failure for required fields
                return RepairResult(
                    success=False,
                    data=None,
                    error_ratio=total_error_ratio + nested_result.error_ratio,
                    errors=errors + nested_result.errors,
                    repaired_errors=repaired_errors + nested_result.repaired_errors,
                    unrepaired_errors=unrepaired_errors + nested_result.unrepaired_errors,
                )

        # Handle lists of objects
        elif key in list_item_schemas and isinstance(value, list):
            repaired_list = []
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    item_path = f"{_path}.{key}[{i}]" if _path else f"{key}[{i}]"
                    item_result = _repair_keys(
                        item,
                        list_item_schemas[key],
                        max_error_ratio_per_key,
                        max_total_error_ratio,
                        strict_validation,
                        drop_unrepairable_items,
                        item_path,
                    )
                    # Propagate or drop item if it fails repair
                    if not item_result.success:
                        if drop_unrepairable_items:
                            continue  # Skip this item
                        else:
                            return RepairResult(
                                success=False,
                                data=None,
                                error_ratio=total_error_ratio + item_result.error_ratio,
                                errors=errors + item_result.errors,
                                repaired_errors=repaired_errors + item_result.repaired_errors,
                                unrepaired_errors=unrepaired_errors + item_result.unrepaired_errors,
                            )
                    repaired_list.append(item_result.data)
                    total_error_ratio += item_result.error_ratio
                    errors.extend(item_result.errors)
                    repaired_errors.extend(item_result.repaired_errors)
                    unrepaired_errors.extend(item_result.unrepaired_errors)
                else:
                    repaired_list.append(item)

            # Check minItems constraint if we dropped items
            if drop_unrepairable_items and not _check_min_items_constraint(
                repaired_list, value, key, json_schema
            ):
                return RepairResult(
                    success=False,
                    data=None,
                    error_ratio=total_error_ratio,
                    errors=errors,
                    repaired_errors=repaired_errors,
                    unrepaired_errors=unrepaired_errors,
                )

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

                repaired_error = RepairError(
                    error_type=ErrorType.misspelled_key,
                    from_key=input_key,
                    to_key=matched_key,
                    error_ratio=error_ratio,
                    path=_path,
                )
                errors.append(repaired_error)
                repaired_errors.append(repaired_error)
                total_error_ratio += error_ratio

                # Handle nested objects for repaired keys
                if matched_key in nested_schemas and isinstance(value, dict):
                    nested_path = f"{_path}.{matched_key}" if _path else matched_key
                    nested_result = _repair_keys(
                        value,
                        nested_schemas[matched_key],
                        max_error_ratio_per_key,
                        max_total_error_ratio,
                        strict_validation,
                        drop_unrepairable_items,
                        nested_path,
                    )
                    if nested_result.success:
                        repaired_data[matched_key] = nested_result.data
                        total_error_ratio += nested_result.error_ratio
                        errors.extend(nested_result.errors)
                        repaired_errors.extend(nested_result.repaired_errors)
                        unrepaired_errors.extend(nested_result.unrepaired_errors)
                    elif drop_unrepairable_items and matched_key not in required_keys:
                        # Drop optional nested object that failed repair
                        # Undo the key repair we just recorded, but keep nested errors for context
                        matched_expected_keys.discard(matched_key)
                        errors.pop()  # Remove key repair error
                        repaired_errors.pop()  # Remove key repair error
                        total_error_ratio -= error_ratio
                        # Add nested errors to show why it was dropped
                        errors.extend(nested_result.errors)
                        repaired_errors.extend(
                            nested_result.errors
                        )  # Treated as "repaired" by dropping
                        continue
                    else:
                        # Propagate nested repair failure for required fields
                        return RepairResult(
                            success=False,
                            data=None,
                            error_ratio=total_error_ratio + nested_result.error_ratio,
                            errors=errors + nested_result.errors,
                            repaired_errors=repaired_errors + nested_result.repaired_errors,
                            unrepaired_errors=unrepaired_errors + nested_result.unrepaired_errors,
                        )

                # Handle lists of objects for repaired keys
                elif matched_key in list_item_schemas and isinstance(value, list):
                    repaired_list = []
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            item_path = (
                                f"{_path}.{matched_key}[{i}]" if _path else f"{matched_key}[{i}]"
                            )
                            item_result = _repair_keys(
                                item,
                                list_item_schemas[matched_key],
                                max_error_ratio_per_key,
                                max_total_error_ratio,
                                strict_validation,
                                drop_unrepairable_items,
                                item_path,
                            )
                            # Propagate or drop item if it fails repair
                            if not item_result.success:
                                if drop_unrepairable_items:
                                    continue  # Skip this item
                                else:
                                    return RepairResult(
                                        success=False,
                                        data=None,
                                        error_ratio=total_error_ratio + item_result.error_ratio,
                                        errors=errors + item_result.errors,
                                        repaired_errors=repaired_errors
                                        + item_result.repaired_errors,
                                        unrepaired_errors=unrepaired_errors
                                        + item_result.unrepaired_errors,
                                    )
                            repaired_list.append(item_result.data)
                            total_error_ratio += item_result.error_ratio
                            errors.extend(item_result.errors)
                            repaired_errors.extend(item_result.repaired_errors)
                            unrepaired_errors.extend(item_result.unrepaired_errors)
                        else:
                            repaired_list.append(item)

                    # Check minItems constraint if we dropped items
                    if drop_unrepairable_items and not _check_min_items_constraint(
                        repaired_list, value, matched_key, json_schema
                    ):
                        return RepairResult(
                            success=False,
                            data=None,
                            error_ratio=total_error_ratio,
                            errors=errors,
                            repaired_errors=repaired_errors,
                            unrepaired_errors=unrepaired_errors,
                        )

                    repaired_data[matched_key] = repaired_list

                # Primitive value
                else:
                    repaired_data[matched_key] = value
            else:
                # No good match
                if drop_unrepairable_items:
                    # Drop the key entirely when enabled, but record it
                    dropped_error = RepairError(
                        error_type=ErrorType.unrecognized_key,
                        from_key=input_key,
                        error_ratio=1.0,
                        message=f"Key '{input_key}' dropped (unrecognized, no close match)",
                        path=_path,
                    )
                    errors.append(dropped_error)
                    repaired_errors.append(dropped_error)
                    continue

                if not strict_validation:
                    repaired_data[input_key] = value

                unrepaired_error = RepairError(
                    error_type=ErrorType.unrecognized_key,
                    from_key=input_key,
                    error_ratio=1.0,
                    message=f"Key '{input_key}' not recognized and no close match found",
                    path=_path,
                )
                errors.append(unrepaired_error)
                unrepaired_errors.append(unrepaired_error)

    # Check for missing required fields
    missing_keys = required_keys - matched_expected_keys
    for missing_key in missing_keys:
        missing_error = RepairError(
            error_type=ErrorType.missing_expected_key,
            to_key=missing_key,
            error_ratio=1.0,
            message=f"Required key '{missing_key}' is missing from input",
            path=_path,
        )
        errors.append(missing_error)
        unrepaired_errors.append(missing_error)

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
                return RepairResult(
                    success=False,
                    data=None,
                    error_ratio=total_error_ratio,
                    errors=errors,
                    repaired_errors=repaired_errors,
                    unrepaired_errors=unrepaired_errors,
                )

        # Reject if average error is too high
        if avg_error_ratio > max_total_error_ratio:
            return RepairResult(
                success=False,
                data=None,
                error_ratio=total_error_ratio,
                errors=errors,
                repaired_errors=repaired_errors,
                unrepaired_errors=unrepaired_errors,
            )

    # In strict mode, reject unrecognized keys
    if strict_validation:
        has_unrecognized = any(e.error_type == ErrorType.unrecognized_key for e in errors)
        if has_unrecognized:
            return RepairResult(
                success=False,
                data=None,
                error_ratio=total_error_ratio,
                errors=errors,
                repaired_errors=repaired_errors,
                unrepaired_errors=unrepaired_errors,
            )

    if unrepaired_errors:
        return RepairResult(
            success=False,
            data=None,
            error_ratio=total_error_ratio,
            errors=errors,
            repaired_errors=repaired_errors,
            unrepaired_errors=unrepaired_errors,
        )

    return RepairResult(
        success=True,
        data=repaired_data,
        error_ratio=total_error_ratio,
        errors=errors,
        repaired_errors=repaired_errors,
        unrepaired_errors=unrepaired_errors,
    )


def fuzzy_model_validate_json(
    json_data: str,
    model_cls: type[BaseModel],
    max_error_ratio_per_key: float = 0.3,
    max_total_error_ratio: float = 0.3,
    strict_validation: bool = False,
    drop_unrepairable_items: bool = False,
) -> Any:
    """
    Repair LLM response JSON data to match a Pydantic model schema.

    This is a convenience function that:
    1. Repairs JSON syntax errors automatically
    2. Tries fast path validation
    3. Falls back to fuzzy key repair if needed
    4. Validates acceptability of repairs
    5. Returns validated Pydantic instance

    Args:
        json_data: JSON string to repair and validate
        model_cls: Pydantic model class to match against
        max_error_ratio_per_key: Maximum allowed error ratio per individual key (0.0-1.0)
        max_total_error_ratio: Maximum average error ratio across all fields (0.0-1.0)
        strict_validation: If True, fail on any unrecognized keys
        drop_unrepairable_items: If True, drop list items, unrecognized keys, and optional
            nested objects that can't be repaired (respects minItems, preserves required fields)

    Returns:
        Validated Pydantic BaseModel instance

    Raises:
        RepairFailedError: If repair fails or validation fails after repair.
            Provides access to repair attempt details via exception.result

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
    try:
        from pydantic import BaseModel, TypeAdapter, ValidationError
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        msg = (
            "fuzzy_model_validate_json requires Pydantic. Install the optional dependency with "
            "`pip install fuzzy-json-repair[pydantic]` or add `pydantic` to your project."
        )
        raise ImportError(msg) from exc

    if not issubclass(model_cls, BaseModel):
        raise TypeError(
            "model_cls must be a Pydantic BaseModel subclass when calling fuzzy_model_validate_json"
        )

    # Repair JSON syntax (always)
    repaired_json = json_repair.repair_json(json_data, skip_json_loads=True)

    # Try fast path first with strict validation to catch unknown fields
    try:
        # Use strict=True to detect unknown fields that might be typos
        return model_cls.model_validate_json(repaired_json, strict=True)
    except Exception:
        # Strict validation failed - likely due to unknown fields or other issues
        # Fall back to key repair to handle typos and preserve data
        pass

    adapter = TypeAdapter(dict[str, Any])

    # Parse JSON using Pydantic's Rust-powered parser
    try:
        parsed_data = adapter.validate_json(repaired_json)
    except ValidationError as e:
        raise ValueError(f"JSON syntax error: {e}") from e

    if not isinstance(parsed_data, dict):
        raise ValueError(f"Expected dict, got {type(parsed_data)}")

    # Get schema and repair keys
    schema = model_cls.model_json_schema()
    result = repair_keys(
        parsed_data,
        schema,
        max_error_ratio_per_key,
        max_total_error_ratio,
        strict_validation,
        drop_unrepairable_items,
    )

    # Check if repair succeeded
    if result.success:
        try:
            # Validate the repaired data
            return model_cls.model_validate(result.data)
        except Exception as validation_error:
            # Repair succeeded but validation failed
            raise ValueError(
                f"Validation failed after key repair for {model_cls.__name__}: "
                f"{str(validation_error)}"
            ) from validation_error

    # Repair failed - create detailed error message
    error_summary = []
    if result.misspelled_keys:
        error_summary.append(f"{len(result.misspelled_keys)} misspelled keys")
    if result.missing_keys:
        error_summary.append(f"{len(result.missing_keys)} missing keys")
    if result.unrecognized_keys:
        error_summary.append(f"{len(result.unrecognized_keys)} unrecognized keys")

    raise RepairFailedError(
        f"JSON repair failed for {model_cls.__name__}. "
        f"Issues found: {', '.join(error_summary)}. "
        f"Details: {[str(e) for e in result.errors]}",
        result=result,
        model_cls=model_cls,
        json_data=json_data,
    )
