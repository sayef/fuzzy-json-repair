"""
Comprehensive tests for fuzzy-json-repair.
"""

import importlib
import json
import sys
import unittest
from unittest import mock

from pydantic import BaseModel

from fuzzy_json_repair import (
    ErrorType,
    fuzzy_model_validate_json,
    repair_keys,
)


# Test Models
class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str


class ContactInfo(BaseModel):
    email: str
    phone: str


class Product(BaseModel):
    product_id: str
    name: str
    price: float
    quantity: int
    in_stock: bool = True


class Customer(BaseModel):
    customer_id: str
    first_name: str
    last_name: str
    age: int
    email: str
    contact: ContactInfo


class Order(BaseModel):
    order_id: str
    customer: Customer
    products: list[Product]
    shipping_address: Address
    total_amount: float


class TestRepairKeys(unittest.TestCase):
    """Test repair_keys function."""

    def test_simple_typos(self):
        """Test simple key typos."""

        class User(BaseModel):
            name: str
            age: int
            email: str

        schema = User.model_json_schema()
        data = {"nam": "John", "agge": 30, "emal": "john@example.com"}

        result = repair_keys(data, schema, max_error_ratio_per_key=0.5)

        self.assertTrue(result.success)
        self.assertEqual(result.data["name"], "John")
        self.assertEqual(result.data["age"], 30)
        self.assertEqual(result.data["email"], "john@example.com")
        self.assertEqual(len(result.errors), 3)
        self.assertEqual(len(result.repaired_errors), 3)
        self.assertEqual(len(result.unrepaired_errors), 0)
        self.assertGreater(result.error_ratio, 0)

    def test_nested_objects(self):
        """Test nested object repair."""

        class User(BaseModel):
            name: str
            address: Address

        schema = User.model_json_schema()
        data = {
            "nam": "John",
            "addres": {"stret": "123 Main", "cty": "NYC", "stat": "NY", "zip_cod": "10001"},
        }

        result = repair_keys(data, schema, max_error_ratio_per_key=0.5)

        self.assertTrue(result.success)
        self.assertEqual(result.data["name"], "John")
        self.assertEqual(result.data["address"]["street"], "123 Main")
        self.assertEqual(result.data["address"]["city"], "NYC")
        self.assertEqual(result.data["address"]["state"], "NY")
        self.assertEqual(result.data["address"]["zip_code"], "10001")
        self.assertTrue(all(e in result.repaired_errors for e in result.errors))
        self.assertEqual(len(result.unrepaired_errors), 0)

    def test_list_of_objects(self):
        """Test lists of objects."""

        class Cart(BaseModel):
            cart_id: str
            products: list[Product]

        schema = Cart.model_json_schema()
        data = {
            "cart_idd": "C123",
            "prodcts": [
                {
                    "product_idd": "P1",
                    "nam": "Laptop",
                    "pric": 999.99,
                    "quantty": 1,
                    "in_stck": True,
                },
                {"product_idd": "P2", "nam": "Mouse", "pric": 29.99, "quantty": 2},
            ],
        }

        result = repair_keys(data, schema, max_error_ratio_per_key=0.5)

        self.assertTrue(result.success)
        self.assertEqual(result.data["cart_id"], "C123")
        self.assertEqual(len(result.data["products"]), 2)
        self.assertEqual(result.data["products"][0]["name"], "Laptop")
        self.assertEqual(result.data["products"][1]["name"], "Mouse")
        self.assertEqual(len(result.unrepaired_errors), 0)
        self.assertGreater(len(result.repaired_errors), 0)

    def test_complex_nested_structure(self):
        """Test very complex nested structure with multiple levels."""
        schema = Order.model_json_schema()

        data = {
            "order_idd": "ORD-12345",
            "custmer": {
                "customer_idd": "C-001",
                "first_nam": "John",
                "last_nam": "Doe",
                "agge": 35,
                "emal": "john@example.com",
                "contct": {"emal": "john.doe@email.com", "phne": "+1-555-0100"},
            },
            "prodcts": [
                {
                    "product_idd": "P-001",
                    "nam": "Laptop",
                    "pric": 1299.99,
                    "quantty": 1,
                    "in_stck": True,
                },
                {
                    "product_idd": "P-002",
                    "nam": "Mouse",
                    "pric": 29.99,
                    "quantty": 2,
                    "in_stck": True,
                },
            ],
            "shipping_addres": {
                "stret": "123 Main Street",
                "cty": "New York",
                "stat": "NY",
                "zip_cod": "10001",
            },
            "total_amnt": 1359.97,
        }

        result = repair_keys(data, schema, max_error_ratio_per_key=0.5)

        self.assertTrue(result.success)
        # Verify root level
        self.assertEqual(result.data["order_id"], "ORD-12345")
        self.assertEqual(result.data["total_amount"], 1359.97)

        # Verify nested customer
        self.assertEqual(result.data["customer"]["customer_id"], "C-001")
        self.assertEqual(result.data["customer"]["first_name"], "John")
        self.assertEqual(result.data["customer"]["last_name"], "Doe")
        self.assertEqual(result.data["customer"]["age"], 35)

        # Verify double-nested contact
        self.assertEqual(result.data["customer"]["contact"]["email"], "john.doe@email.com")
        self.assertEqual(result.data["customer"]["contact"]["phone"], "+1-555-0100")

        # Verify list of products
        self.assertEqual(len(result.data["products"]), 2)
        self.assertEqual(result.data["products"][0]["product_id"], "P-001")
        self.assertEqual(result.data["products"][0]["name"], "Laptop")

        # Verify shipping address
        self.assertEqual(result.data["shipping_address"]["street"], "123 Main Street")
        self.assertEqual(result.data["shipping_address"]["city"], "New York")
        self.assertEqual(len(result.unrepaired_errors), 0)
        self.assertEqual(len(result.repaired_errors), len(result.errors))

    def test_error_types(self):
        """Test different error types."""

        class User(BaseModel):
            name: str
            age: int

        schema = User.model_json_schema()
        data = {
            "nam": "John",  # misspelled
            "unknown_field": "value",  # unrecognized
            # 'age' is missing
        }

        result = repair_keys(data, schema, max_error_ratio_per_key=0.5)

        self.assertFalse(result.success)
        error_types = {e.error_type for e in result.errors}
        self.assertIn(ErrorType.misspelled_key, error_types)
        self.assertIn(ErrorType.unrecognized_key, error_types)
        self.assertIn(ErrorType.missing_expected_key, error_types)
        self.assertEqual(len(result.repaired_errors), 1)
        self.assertEqual(len(result.unrepaired_errors), 2)

    def test_drop_unrepairable_keys(self):
        """Test dropping of unrecognized keys that can't be matched."""

        class User(BaseModel):
            name: str
            age: int

        schema = User.model_json_schema()
        data = {"name": "John", "age": 30, "unknown_field": "value", "another_bad_key": 123}

        result = repair_keys(data, schema, drop_unrepairable_items=True)

        self.assertTrue(result.success)
        self.assertEqual(result.data, {"name": "John", "age": 30})
        self.assertNotIn("unknown_field", result.data)
        self.assertNotIn("another_bad_key", result.data)
        self.assertEqual(len(result.unrepaired_errors), 0)
        # Both dropped keys should be recorded
        self.assertEqual(len(result.repaired_errors), 2)
        dropped_keys = {e.from_key for e in result.repaired_errors}
        self.assertEqual(dropped_keys, {"unknown_field", "another_bad_key"})

    def test_drop_unrepairable_keys_with_typos(self):
        """Test that repairable typos are fixed and unrepairable keys are dropped."""

        class User(BaseModel):
            name: str
            age: int
            email: str

        schema = User.model_json_schema()
        data = {"nam": "John", "age": 30, "emal": "john@example.com", "completely_wrong": "value"}

        result = repair_keys(data, schema, drop_unrepairable_items=True)

        self.assertTrue(result.success)
        self.assertEqual(result.data["name"], "John")
        self.assertEqual(result.data["age"], 30)
        self.assertEqual(result.data["email"], "john@example.com")
        self.assertNotIn("completely_wrong", result.data)
        # 3 repaired: nam->name, emal->email, completely_wrong dropped
        self.assertEqual(len(result.repaired_errors), 3)
        self.assertEqual(len(result.unrepaired_errors), 0)
        # Verify dropped key is recorded
        dropped = [
            e for e in result.repaired_errors if e.message and "dropped" in e.message.lower()
        ]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].from_key, "completely_wrong")

    def test_drop_unrepairable_optional_nested_object(self):
        """Test dropping of optional nested objects that fail repair."""

        class Profile(BaseModel):
            nickname: str
            bio: str

        class User(BaseModel):
            name: str
            age: int
            profile: Profile | None = None

        schema = User.model_json_schema()
        data = {"name": "John", "age": 30, "profile": {"unknown": "value", "bad": "data"}}

        result = repair_keys(data, schema, drop_unrepairable_items=True)

        self.assertTrue(result.success)
        self.assertEqual(result.data["name"], "John")
        self.assertEqual(result.data["age"], 30)
        self.assertNotIn("profile", result.data)
        self.assertEqual(len(result.unrepaired_errors), 0)
        # Optional nested object drop is recorded in errors (from nested repair attempt)
        self.assertGreater(len(result.errors), 0)

    def test_cannot_drop_required_nested_object(self):
        """Test that required nested objects that fail repair cause failure."""

        class Profile(BaseModel):
            nickname: str

        class User(BaseModel):
            name: str
            profile: Profile  # Required

        schema = User.model_json_schema()
        data = {"name": "John", "profile": {"unknown": "value"}}

        result = repair_keys(data, schema, drop_unrepairable_items=True)

        self.assertFalse(result.success)
        self.assertIsNone(result.data)
        self.assertGreater(len(result.unrepaired_errors), 0)

    def test_drop_optional_with_typo_in_key(self):
        """Test dropping optional nested object when the key itself has a typo."""

        class Profile(BaseModel):
            nickname: str

        class User(BaseModel):
            name: str
            profile: Profile | None = None

        schema = User.model_json_schema()
        data = {"name": "John", "profle": {"unknown": "value"}}  # Typo in 'profile'

        result = repair_keys(data, schema, drop_unrepairable_items=True)

        self.assertTrue(result.success)
        self.assertEqual(result.data["name"], "John")
        self.assertNotIn("profile", result.data)
        self.assertEqual(len(result.unrepaired_errors), 0)
        # Should have errors from key repair attempt and nested repair attempt
        self.assertGreater(len(result.errors), 0)


class TestFuzzyModelValidateJson(unittest.TestCase):
    """Test fuzzy_model_validate_json function."""

    def test_simple_validation(self):
        """Test simple JSON validation."""

        class User(BaseModel):
            name: str
            age: int

        json_str = '{"nam": "John", "agge": 30}'
        user = fuzzy_model_validate_json(json_str, User, max_total_error_ratio=1.0)

        self.assertIsInstance(user, User)
        self.assertEqual(user.name, "John")
        self.assertEqual(user.age, 30)

    def test_fast_path(self):
        """Test that fast path works with correct JSON."""

        class User(BaseModel):
            name: str
            age: int

        json_str = '{"name": "John", "age": 30}'
        user = fuzzy_model_validate_json(json_str, User)

        self.assertEqual(user.name, "John")
        self.assertEqual(user.age, 30)

    def test_nested_validation(self):
        """Test nested object validation."""

        class User(BaseModel):
            name: str
            address: Address

        json_str = json.dumps(
            {
                "nam": "John",
                "addres": {"stret": "123 Main", "cty": "NYC", "stat": "NY", "zip_cod": "10001"},
            }
        )

        user = fuzzy_model_validate_json(json_str, User, max_total_error_ratio=2.0)

        self.assertEqual(user.name, "John")
        self.assertEqual(user.address.street, "123 Main")

    def test_complex_validation(self):
        """Test complex nested validation."""
        json_str = json.dumps(
            {
                "order_idd": "ORD-123",
                "custmer": {
                    "customer_idd": "C-001",
                    "first_nam": "John",
                    "last_nam": "Doe",
                    "agge": 35,
                    "emal": "john@example.com",
                    "contct": {"emal": "john@email.com", "phne": "+1-555-0100"},
                },
                "prodcts": [
                    {"product_idd": "P-001", "nam": "Laptop", "pric": 1299.99, "quantty": 1}
                ],
                "shipping_addres": {
                    "stret": "123 Main",
                    "cty": "NYC",
                    "stat": "NY",
                    "zip_cod": "10001",
                },
                "total_amnt": 1299.99,
            }
        )

        order = fuzzy_model_validate_json(json_str, Order, max_total_error_ratio=3.0)

        self.assertIsInstance(order, Order)
        self.assertEqual(order.order_id, "ORD-123")
        self.assertEqual(order.customer.first_name, "John")
        self.assertEqual(len(order.products), 1)
        self.assertEqual(order.products[0].name, "Laptop")

    def test_validation_failure(self):
        """Test that validation fails for too many result.errors."""
        from fuzzy_json_repair import RepairFailedError

        class User(BaseModel):
            name: str
            age: int

        # Completely wrong data
        json_str = '{"wrong": "data", "bad": "keys"}'

        with self.assertRaises(RepairFailedError) as cm:
            fuzzy_model_validate_json(json_str, User, max_total_error_ratio=0.3)

        # Should fail with validation error
        self.assertIn("failed", str(cm.exception).lower())

        # Check structured access to error details
        error = cm.exception
        self.assertIsNotNone(error.result)
        self.assertEqual(error.model_cls, User)
        self.assertEqual(error.json_data, json_str)
        self.assertGreater(len(error.errors), 0)
        self.assertGreater(len(error.unrepaired_errors), 0)


class TestRepairThresholds(unittest.TestCase):
    """Test that repair_keys returns None when thresholds are exceeded."""

    def test_acceptable_repair(self):
        """Test that good repairs return data."""

        class User(BaseModel):
            name: str
            age: int
            email: str

        schema = User.model_json_schema()
        data = {"nam": "John", "agge": 30, "emal": "john@example.com"}

        result = repair_keys(data, schema, max_error_ratio_per_key=0.5, max_total_error_ratio=0.3)

        # Should succeed with good typos
        self.assertTrue(result.success)
        self.assertEqual(len(result.errors), 3)
        self.assertEqual(len(result.repaired_errors), 3)
        self.assertEqual(len(result.unrepaired_errors), 0)

    def test_excessive_total_error_ratio(self):
        """Test that repairs with high average error return None."""

        class User(BaseModel):
            name: str
            age: int
            email: str

        schema = User.model_json_schema()
        # Multiple typos that sum to high total error
        data = {"nme": "John", "ag": 30, "eml": "john@example.com"}

        result = repair_keys(data, schema, max_error_ratio_per_key=0.5, max_total_error_ratio=0.1)

        # Should fail due to high total error result.error_ratio (3 typos / 3 fields > 0.1)
        self.assertFalse(result.success)
        self.assertGreater(result.error_ratio, 0)
        self.assertEqual(len(result.unrepaired_errors), 0)
        self.assertEqual(len(result.repaired_errors), len(result.errors))

    def test_strict_mode_rejects_unrecognized(self):
        """Test strict mode returns None for unrecognized keys."""

        class User(BaseModel):
            name: str
            age: int

        schema = User.model_json_schema()
        data = {"name": "John", "age": 30, "unknown_field": "value"}

        result = repair_keys(data, schema, max_error_ratio_per_key=0.5, strict_validation=True)

        # Should fail in strict mode with unrecognized keys
        self.assertFalse(result.success)
        has_unrecognized = any(e.error_type == ErrorType.unrecognized_key for e in result.errors)
        self.assertTrue(has_unrecognized)
        self.assertGreater(len(result.unrepaired_errors), 0)
        self.assertTrue(all(e in result.unrepaired_errors for e in result.unrecognized_keys))


class TestOptionalPydantic(unittest.TestCase):
    """Ensure Pydantic is only required for high-level API."""

    def test_repair_keys_import_without_pydantic(self):
        original_pydantic = sys.modules.pop("pydantic", None)
        with mock.patch.dict(sys.modules, {"pydantic": None}):
            import fuzzy_json_repair
            import fuzzy_json_repair.repair as repair_module

            importlib.reload(repair_module)
            importlib.reload(fuzzy_json_repair)

            result = repair_module.repair_keys({}, {"properties": {}})
            self.assertTrue(result.success)
            self.assertEqual(result.unrepaired_errors, [])

            with self.assertRaises(ImportError):
                repair_module.fuzzy_model_validate_json("{}", type("Dummy", (), {}))

        if original_pydantic is not None:
            sys.modules["pydantic"] = original_pydantic
        import fuzzy_json_repair.repair as repair_module

        importlib.reload(repair_module)
        import fuzzy_json_repair  # noqa: F401

        importlib.reload(fuzzy_json_repair)


if __name__ == "__main__":
    unittest.main()
