"""
Comprehensive tests for fuzzy-json-repair.
"""

import json
import unittest

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

        print(f"\n✅ Complex nested: {len(result.errors)} typos repaired")
        print(f"   Total error result.error_ratio: {result.error_ratio:.2%}")

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

        error_types = {e.error_type for e in result.errors}
        self.assertIn(ErrorType.misspelled_key, error_types)
        self.assertIn(ErrorType.unrecognized_key, error_types)
        self.assertIn(ErrorType.missing_expected_key, error_types)


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

        print("\n✅ Complex validation successful!")

    def test_validation_failure(self):
        """Test that validation fails for too many result.errors."""

        class User(BaseModel):
            name: str
            age: int

        # Completely wrong data
        json_str = '{"wrong": "data", "bad": "keys"}'

        with self.assertRaises(ValueError) as cm:
            fuzzy_model_validate_json(json_str, User, max_total_error_ratio=0.3)

        # Should fail with validation error
        self.assertIn("failed", str(cm.exception).lower())


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


if __name__ == "__main__":
    unittest.main()
