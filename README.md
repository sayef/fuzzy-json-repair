# Fuzzy JSON Repair

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

You ask an LLM for JSON, it gives you `{"nam": "John", "emal": "john@example.com"}` instead of `{"name": "John", "email": "john@example.com"}`. Your Pydantic validation fails. You spend an hour writing error handling code.

This library fixes those typos automatically using fuzzy string matching. No more manual key mapping, no more try-except blocks everywhere.

## Install

```bash
pip install fuzzy-json-repair
```

If you're processing lots of data, install with numpy for ~10x faster batch processing:

```bash
pip install fuzzy-json-repair[fast]
```

## Usage

The simplest way - repair and validate in one go:

```python
from pydantic import BaseModel
from fuzzy_json_repair import fuzzy_model_validate_json

class User(BaseModel):
    name: str
    age: int
    email: str

# Your LLM gave you this
json_str = '{"nam": "John", "agge": 30, "emal": "john@example.com"}'

# This just works
user = fuzzy_model_validate_json(json_str, User)
print(user)  # User(name='John', age=30, email='john@example.com')
```

Or if you want more control:

```python
from fuzzy_json_repair import repair_keys

schema = User.model_json_schema()
data = {'nam': 'John', 'agge': 30, 'emal': 'john@example.com'}

result = repair_keys(data, schema)

if result.success:
    user = User.model_validate(result.data)
else:
    print(f"Repair failed: {len(result.errors)} errors")
    print(f"Error ratio: {result.error_ratio:.2%}")
```

## Advanced Usage

### Nested Objects

```python
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Person(BaseModel):
    name: str
    address: Address

data = {
    'nam': 'John',
    'addres': {
        'stret': '123 Main St',
        'cty': 'NYC',
        'zip_cod': '10001'
    }
}

schema = Person.model_json_schema()
result = repair_keys(data, schema, max_error_ratio_per_key=0.5)

# All nested typos are fixed!
if result.success:
    person = Person.model_validate(result.data)
```

### Lists of Objects

```python
class Product(BaseModel):
    product_id: str
    name: str
    price: float

class Cart(BaseModel):
    cart_id: str
    products: list[Product]

data = {
    'cart_idd': 'C123',
    'prodcts': [
        {'product_idd': 'P1', 'nam': 'Laptop', 'pric': 999.99},
        {'product_idd': 'P2', 'nam': 'Mouse', 'pric': 29.99}
    ]
}

schema = Cart.model_json_schema()
result = repair_keys(data, schema, max_error_ratio_per_key=0.5)

# Repairs all typos in the list items too!
if result.success:
    cart = Cart.model_validate(result.data)
```

### Drop Unrepairable Items

Sometimes list items are beyond repair. Drop them automatically while respecting `minItems` constraints:

```python
class Product(BaseModel):
    name: str
    price: float

class Cart(BaseModel):
    items: list[Product]

data = {
    'items': [
        {'nam': 'Laptop', 'pric': 999},        # Repairable
        {'completely': 'wrong', 'keys': 123},  # Beyond repair
        {'nme': 'Mouse', 'prce': 29}           # Repairable
    ]
}

schema = Cart.model_json_schema()
result = repair_keys(
    data, schema,
    drop_unrepairable_items=True  # Drop items that can't be fixed
)

if result.success:
    # Returns 2 items (dropped the broken one)
    print(len(result.data['items']))  # 2
```

Works with nested structures too:

```python
class Order(BaseModel):
    order_id: str
    products: list[Product]

class Customer(BaseModel):
    name: str
    orders: list[Order]

# Drops unrepairable items at any nesting level
result = repair_keys(
    data, schema,
    drop_unrepairable_items=True
)
if result.success:
    use(result.data)
```

### Complex Nested Structures

```python
class Customer(BaseModel):
    customer_id: str
    name: str
    email: str

class Order(BaseModel):
    order_id: str
    customer: Customer
    products: list[Product]
    total: float

# Works with arbitrarily complex nesting!
json_str = '''
{
    "order_idd": "ORD-123",
    "custmer": {
        "customer_idd": "C-001",
        "nam": "John",
        "emal": "john@example.com"
    },
    "prodcts": [
        {"product_idd": "P-001", "nam": "Laptop", "pric": 1299.99}
    ],
    "totl": 1299.99
}
'''

order = fuzzy_model_validate_json(
    json_str,
    Order,
    max_total_error_ratio=2.0  # Allow higher error ratio for complex structures
)
```

## API Reference

### `repair_keys(data, json_schema, max_error_ratio_per_key=0.3, max_total_error_ratio=0.5, strict_validation=False, drop_unrepairable_items=False)`

Repair dictionary keys using fuzzy matching against a JSON schema.

**Parameters:**
- `data` (dict): Input dictionary with potential typos
- `json_schema` (dict): JSON schema from `model.model_json_schema()`
- `max_error_ratio_per_key` (float): Maximum error ratio per individual key (0.0-1.0). Default: 0.3
- `max_total_error_ratio` (float): Maximum average error ratio across all schema fields (0.0-1.0). Default: 0.5
- `strict_validation` (bool): If True, reject unrecognized keys. Default: False
- `drop_unrepairable_items` (bool): If True, drop list items that can't be repaired (respects minItems). Default: False

**Returns:**
- `RepairResult`: Object with:
  - `success` (bool): Whether repair succeeded
  - `data` (dict | None): Repaired data (None if failed)
  - `error_ratio` (float): Total error ratio
  - `errors` (list[RepairError]): List of errors encountered

**Example:**
```python
schema = User.model_json_schema()
result = repair_keys(data, schema)
if result.success:
    user = User.model_validate(result.data)
else:
    print(f"Repair failed: {len(result.errors)} errors")
```

### `fuzzy_model_validate_json(json_data, model_cls, repair_syntax=True, max_error_ratio_per_key=0.3, max_total_error_ratio=0.3, strict_validation=False, drop_unrepairable_items=False)`

Repair JSON string and return validated Pydantic model instance.

**Parameters:**
- `json_data` (str): JSON string to repair
- `model_cls` (type[BaseModel]): Pydantic model class
- `repair_syntax` (bool): Attempt to fix JSON syntax errors. Default: True (requires json-repair)
- `max_error_ratio_per_key` (float): Max error per individual key. Default: 0.3
- `max_total_error_ratio` (float): Max average error across all fields. Default: 0.3
- `strict_validation` (bool): Reject unrecognized keys. Default: False
- `drop_unrepairable_items` (bool): Drop list items that can't be repaired (respects minItems). Default: False

**Returns:**
- `BaseModel`: Validated Pydantic model instance

**Raises:**
- `ValueError`: If repair fails or validation fails

**Example:**
```python
user = fuzzy_model_validate_json(json_str, User)
```

## Error Types

```python
from fuzzy_json_repair import ErrorType, RepairError, RepairResult

# ErrorType enum:
ErrorType.misspelled_key       # Typo was fixed
ErrorType.unrecognized_key     # Unknown key (kept if not strict)
ErrorType.missing_expected_key  # Required field missing

# RepairError dataclass:
error = RepairError(
    error_type=ErrorType.misspelled_key,
    from_key='nam',
    to_key='name',
    error_ratio=0.143,
)
print(error)
# "Misspelled key 'nam' → 'name' (error: 14.3%)"

# RepairResult dataclass:
result = RepairResult(
    success=True,
    data={'name': 'John', 'age': 30},
    error_ratio=0.15,
    errors=[error]
)
print(f"Success: {result.success}")
print(f"Misspelled: {len(result.misspelled_keys)}")
print(f"Failed: {result.failed}")
```

## Configuration

### Error Ratio Thresholds

```python
# Strict (only very close matches)
repair_keys(data, schema, max_error_ratio_per_key=0.2)

# Moderate (default, good for most cases)
repair_keys(data, schema, max_error_ratio_per_key=0.3)

# Lenient (fix even poor matches)
repair_keys(data, schema, max_error_ratio_per_key=0.5)
```

### Strict Validation

```python
# Reject unrecognized keys
result = repair_keys(data, schema, strict_validation=True)
if result.success:
    use(result.data)
```

### Drop Unrepairable Items

```python
# Drop list items that exceed error thresholds
result = repair_keys(
    data, schema,
    drop_unrepairable_items=True
)

# Respects minItems constraints
from pydantic import Field

class Cart(BaseModel):
    items: list[Product] = Field(min_length=2)

# If dropping would violate minItems=2, repair fails
result = repair_keys(data, schema, drop_unrepairable_items=True)
if not result.success:
    print("Would violate minItems constraint")
```

## Performance

The library uses two matching strategies:

- **With numpy**: Uses `process.cdist()` for batch processing (10-20x faster)
- **Without numpy**: Uses `process.extractOne()` loop (still fast)

Both use `fuzz.ratio` from RapidFuzz - no raw Levenshtein distance anywhere.

**Benchmark (1000 repairs):**
- With numpy: ~0.05s
- Without numpy: ~0.5s

Install with `pip install fuzzy-json-repair[fast]` for best performance.

## How It Works

1. **Schema Extraction**: Extracts expected keys, nested schemas, and `$ref` definitions from Pydantic's JSON schema
2. **Exact Matching**: Processes keys that match exactly (fast path)
3. **Fuzzy Matching**: For typos, uses RapidFuzz's `fuzz.ratio` to find best match
4. **Batch Processing**: Computes all similarities at once with `cdist` (when numpy available)
5. **Recursive Repair**: Automatically handles nested objects and lists
6. **Validation**: Returns repaired data ready for Pydantic validation

## Use Cases

- **LLM Output Validation**: Fix typos in JSON generated by language models
- **API Integration**: Handle variations in third-party API responses
- **Data Migration**: Repair legacy data with inconsistent field names
- **User Input**: Correct typos in user-provided configuration files
- **Robust Parsing**: Build fault-tolerant JSON parsers

## Requirements

- Python 3.11+
- pydantic >= 2.0.0
- rapidfuzz >= 3.0.0

**Optional:**
- numpy >= 1.20.0 (for faster batch processing)
- json-repair >= 0.7.0 (for JSON syntax repair)

## Development

```bash
# Clone repository
git clone https://github.com/sayef/fuzzy-json-repair.git
cd fuzzy-json-repair

# Install with dev dependencies
pip install -e ".[dev,fast,syntax]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=fuzzy_json_repair --cov-report=term-missing

# Format code
black fuzzy_json_repair tests
isort fuzzy_json_repair tests

# Type check
mypy fuzzy_json_repair

# Lint
ruff check fuzzy_json_repair tests
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits

- Uses [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) for fast fuzzy matching
- Built for [Pydantic](https://github.com/pydantic/pydantic) integration
- Optional [json-repair](https://github.com/mangiucugna/json_repair) support
