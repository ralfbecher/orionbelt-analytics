"""Security tests for the Database Ontology MCP Server."""

import unittest
from unittest.mock import patch, MagicMock

from src.security import (
    SQLInjectionValidator,
    IdentifierValidator,
    SecureCredentialManager,
    SecurityLevel,
    audit_log_security_event
)
from src.database_manager import DatabaseManager


class TestSQLInjectionValidator(unittest.TestCase):
    """Test suite for SQL injection prevention."""

    def setUp(self) -> None:
        self.validator = SQLInjectionValidator()

    def test_safe_select_queries(self) -> None:
        """Test that safe SELECT queries are allowed."""
        safe_queries = [
            "SELECT * FROM users",
            "SELECT id, name FROM users WHERE active = true",
            "SELECT COUNT(*) FROM orders ORDER BY created_at LIMIT 10",
            ("SELECT u.name, o.total FROM users u JOIN orders o ON "
             "u.id = o.user_id"),
        ]

        for query in safe_queries:
            result = self.validator.validate_query(query)
            self.assertTrue(
                result["is_safe"], f"Safe query marked as unsafe: {query}"
            )
            self.assertEqual(result["risk_level"], "low")

    def test_sql_injection_attempts(self) -> None:
        """Test that SQL injection attempts are blocked."""
        malicious_queries = [
            "SELECT * FROM users; DROP TABLE users; --",
            "SELECT * FROM users WHERE id = 1 OR 1=1",
            "SELECT * FROM users UNION SELECT password FROM admin",
            "SELECT * FROM users WHERE name = 'admin'--",
            "SELECT * FROM users; DELETE FROM users WHERE 1=1",
            "SELECT /*comment*/ * FROM users",
            ("SELECT * FROM users WHERE id = 1; INSERT INTO logs "
             "VALUES ('hacked')"),
        ]

        for query in malicious_queries:
            result = self.validator.validate_query(query)
            self.assertFalse(
                result["is_safe"], f"Malicious query not blocked: {query}"
            )
            self.assertIn(result["risk_level"], ["medium", "high", "critical"])
            self.assertTrue(len(result["issues"]) > 0)

    def test_multiple_statements(self) -> None:
        """Test that multiple statements are blocked."""
        multi_statement_queries = [
            "SELECT * FROM users; SELECT * FROM orders",
            "UPDATE users SET name = 'test'; SELECT * FROM users",
            "SELECT 1; SELECT 2; SELECT 3",
        ]

        for query in multi_statement_queries:
            result = self.validator.validate_query(query)
            self.assertFalse(
                result["is_safe"], f"Multiple statements not blocked: {query}"
            )
            self.assertIn(
                "Multiple SQL statements not allowed", result["issues"]
            )

    def test_empty_queries(self) -> None:
        """Test handling of empty or whitespace-only queries."""
        empty_queries = ["", "   ", "\t\n", None]

        for query in empty_queries:
            if query is not None:
                result = self.validator.validate_query(query)
                self.assertFalse(result["is_safe"])
                self.assertEqual(result["risk_level"], "high")

    def test_query_sanitization(self) -> None:
        """Test that queries are properly sanitized for logging."""
        query_with_data = (
            "SELECT * FROM users WHERE name = 'John Doe' AND "
            "email = 'john@example.com'"
        )
        sanitized = self.validator._sanitize_query_for_logging(query_with_data)

        # Should replace string literals with ***
        self.assertIn("'***'", sanitized)
        self.assertNotIn("John Doe", sanitized)
        self.assertNotIn("john@example.com", sanitized)


class TestIdentifierValidator(unittest.TestCase):
    """Test suite for database identifier validation."""

    def test_valid_identifiers(self) -> None:
        """Test that valid identifiers are accepted."""
        valid_identifiers = [
            "users",
            "user_profiles",
            "order_items",
            "UserTable",
            "_private_table",
            "table123",
            "a",
            "TABLE_WITH_UNDERSCORES"
        ]

        for identifier in valid_identifiers:
            self.assertTrue(
                IdentifierValidator.validate_identifier(identifier),
                f"Valid identifier rejected: {identifier}"
            )

    def test_invalid_identifiers(self) -> None:
        """Test that invalid identifiers are rejected."""
        invalid_identifiers = [
            "",  # Empty
            "123table",  # Starts with number
            "table-with-hyphens",  # Contains hyphens (in some contexts)
            "table with spaces",  # Contains spaces
            "table;drop",  # Contains semicolon
            "table'quote",  # Contains quote
            "table/*comment*/",  # Contains comment
            "a" * 200,  # Too long
            "table.schema.database.extra",  # Too many parts
        ]

        for identifier in invalid_identifiers:
            self.assertFalse(
                IdentifierValidator.validate_identifier(identifier),
                f"Invalid identifier accepted: {identifier}"
            )

    def test_qualified_identifiers(self) -> None:
        """Test schema.table identifier validation."""
        valid_qualified = [
            "schema.table",
            "database.schema.table",
            "public.users",
            "_schema._table"
        ]

        invalid_qualified = [
            "schema.table.extra.part",  # Too many parts
            "schema..table",  # Empty part
            ".table",  # Empty schema
            "schema.",  # Empty table
        ]

        for identifier in valid_qualified:
            self.assertTrue(
                IdentifierValidator.validate_qualified_identifier(identifier),
                f"Valid qualified identifier rejected: {identifier}"
            )

        for identifier in invalid_qualified:
            self.assertFalse(
                IdentifierValidator.validate_qualified_identifier(identifier),
                f"Invalid qualified identifier accepted: {identifier}"
            )

    def test_identifier_sanitization(self) -> None:
        """Test identifier sanitization."""
        test_cases = [
            ("table with spaces", "table_with_spaces"),
            ("table-with-hyphens", "table_with_hyphens"),
            ("123table", "_123table"),
            ("table@#$%", "table____"),
            ("", ""),
        ]

        for original, expected in test_cases:
            sanitized = IdentifierValidator.sanitize_identifier(original)
            self.assertEqual(sanitized, expected)


class TestSecureCredentialManager(unittest.TestCase):
    """Test suite for secure credential management."""

    def setUp(self) -> None:
        self.manager = SecureCredentialManager("test_master_password")

    def test_credential_encryption_decryption(self) -> None:
        """Test that credentials can be encrypted and decrypted."""
        test_credentials = {
            "type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "username": "testuser",
            "password": "secret123"
        }

        # Encrypt credentials
        encrypted = self.manager.encrypt_credentials(test_credentials)
        self.assertIsInstance(encrypted, str)
        self.assertNotIn("secret123", encrypted)

        # Decrypt credentials
        decrypted = self.manager.decrypt_credentials(encrypted)
        self.assertEqual(decrypted, test_credentials)

    def test_credential_sanitization(self) -> None:
        """Test that sensitive data is removed from logs."""
        test_credentials = {
            "host": "localhost",
            "username": "testuser",
            "password": "secret123",
            "api_key": "key123",
            "token": "token456"
        }

        sanitized = self.manager._sanitize_credentials(test_credentials)

        # Non-sensitive data should remain
        self.assertEqual(sanitized["host"], "localhost")
        self.assertEqual(sanitized["username"], "testuser")

        # Sensitive data should be redacted
        self.assertEqual(sanitized["password"], "***REDACTED***")
        self.assertEqual(sanitized["api_key"], "***REDACTED***")
        self.assertEqual(sanitized["token"], "***REDACTED***")

    def test_encryption_without_master_password(self) -> None:
        """Test that encryption fails without master password."""
        # Patch load_dotenv and os.getenv to prevent loading from .env file
        with patch('src.security.load_dotenv'), \
             patch('src.security.os.getenv', return_value=None):
            manager = SecureCredentialManager()

            with self.assertRaises(ValueError):
                manager.encrypt_credentials({"password": "test"})

    def test_invalid_decryption_data(self) -> None:
        """Test handling of invalid decryption data."""
        with self.assertRaises(ValueError):
            self.manager.decrypt_credentials("invalid_encrypted_data")


class TestDatabaseManagerSecurity(unittest.TestCase):
    """Test security features of DatabaseManager."""

    def setUp(self) -> None:
        self.db_manager = DatabaseManager()

    @patch('src.database_manager.create_engine')
    def test_secure_postgresql_connection(
        self, mock_create_engine: MagicMock
    ) -> None:
        """Test that PostgreSQL connections use secure practices."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock connection test
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        result = self.db_manager.connect_postgresql(
            host="localhost",
            port=5432,
            database="testdb",
            username="testuser",
            password="testpass"
        )

        self.assertTrue(result)

        # Verify connection string format
        call_args = mock_create_engine.call_args[0][0]
        self.assertIn("postgresql://", call_args)
        # Connection string should include user credentials
        self.assertIn("testuser", call_args)
        self.assertIn("@", call_args)

    def test_identifier_validation_in_methods(self) -> None:
        """Test that database methods validate identifiers."""
        # Set up engine mock to pass the connection check
        self.db_manager.engine = MagicMock()

        # Test with invalid table name (contains semicolon)
        with self.assertRaises(ValueError):
            self.db_manager.sample_table_data("invalid;table", limit=10)

        # Test with invalid schema name (contains semicolon)
        with self.assertRaises(ValueError):
            self.db_manager.sample_table_data(
                "valid_table", "invalid;schema", limit=10
            )

    @patch('src.database_manager.sql_validator')
    def test_sql_validation_integration(
        self, mock_validator: MagicMock
    ) -> None:
        """Test that SQL validation is integrated into query execution."""
        # Set up engine mock to pass the connection check
        self.db_manager.engine = MagicMock()

        # Mock validator to return unsafe query
        mock_validator.validate_query.return_value = {
            "is_safe": False,
            "issues": ["Potential SQL injection"],
            "risk_level": "critical"
        }

        result = self.db_manager.validate_sql_syntax(
            "SELECT * FROM users; DROP TABLE users;"
        )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["error_type"], "security_error")
        self.assertIn("Security validation failed", result["error"])


class TestSecurityAuditing(unittest.TestCase):
    """Test security auditing functionality."""

    @patch('src.security.logger')
    def test_security_event_logging(self, mock_logger: MagicMock) -> None:
        """Test that security events are properly logged."""
        audit_log_security_event(
            "test_security_event",
            {"user": "testuser", "action": "attempted_injection"},
            SecurityLevel.HIGH
        )

        # Verify that warning was logged
        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]

        self.assertIn("SECURITY_AUDIT", log_message)
        self.assertIn("test_security_event", log_message)
        self.assertIn("high", log_message)

    @patch('src.security.logger')
    def test_sensitive_data_redaction_in_audit(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that sensitive data is redacted in audit logs."""
        audit_log_security_event(
            "credential_test",
            {
                "username": "testuser",
                "password": "secret123",
                "host": "localhost"
            },
            SecurityLevel.MEDIUM
        )

        log_message = mock_logger.warning.call_args[0][0]

        # Sensitive data should be redacted
        self.assertNotIn("secret123", log_message)
        self.assertIn("***REDACTED***", log_message)

        # Non-sensitive data should remain
        self.assertIn("testuser", log_message)
        self.assertIn("localhost", log_message)


if __name__ == '__main__':
    unittest.main(verbosity=2)
