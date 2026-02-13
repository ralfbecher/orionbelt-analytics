"""Security utilities for credential management and data protection."""

import base64
import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecureCredentialManager:
    """Secure credential management with encryption."""

    def __init__(self, master_password: Optional[str] = None):
        """Initialize with optional master password for encryption."""
        self._cipher: Optional[Fernet] = None
        self._salt_file = Path.home() / '.mcp_credential_salt'

        # Try to get master password from environment if not provided
        if not master_password:
            load_dotenv()
            master_password = os.getenv('MCP_MASTER_PASSWORD')

        if master_password:
            self._initialize_encryption(master_password)

    def _initialize_encryption(self, master_password: str) -> None:
        """Initialize encryption with master password."""
        # Get or create persistent salt
        salt = self._get_or_create_salt()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        self._cipher = Fernet(key)
        self._salt = salt

    def _get_or_create_salt(self) -> bytes:
        """Get existing salt or create a new one with secure permissions."""
        try:
            if self._salt_file.exists():
                with open(self._salt_file, 'rb') as f:
                    salt = f.read()
                    if len(salt) == 16:  # Validate salt length
                        logger.debug(
                            "Using existing salt for credential encryption"
                        )
                        return salt
                    else:
                        logger.warning("Invalid salt file, creating new salt")

            # Create new salt
            salt = os.urandom(16)

            # Write salt with secure permissions
            with open(self._salt_file, 'wb') as f:
                f.write(salt)

            # Set restrictive permissions (owner read/write only)
            self._salt_file.chmod(0o600)

            logger.info("Created new salt file: %s", self._salt_file)
            return salt

        except (OSError, IOError) as e:
            logger.error("Failed to manage salt file: %s", e)
            # Fallback to session-only salt (less secure)
            logger.warning(
                "Using session-only salt - encrypted data will not persist "
                "across restarts"
            )
            return os.urandom(16)

    def encrypt_credentials(self, credentials: Dict[str, Any]) -> str:
        """Encrypt credentials dictionary."""
        if not self._cipher:
            raise ValueError("Encryption not initialized")

        # Remove sensitive data from logs
        safe_creds = self._sanitize_credentials(credentials)
        logger.debug(
            "Encrypting credentials for %s connection",
            safe_creds.get('type', 'unknown')
        )

        json_data = json.dumps(credentials).encode()
        encrypted_data = self._cipher.encrypt(json_data)
        return base64.urlsafe_b64encode(encrypted_data).decode()

    def decrypt_credentials(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt credentials dictionary."""
        if not self._cipher:
            raise ValueError("Encryption not initialized")

        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._cipher.decrypt(decoded_data)
            credentials_dict: Dict[str, Any] = json.loads(
                decrypted_data.decode()
            )
            return credentials_dict
        except Exception as e:
            logger.error("Failed to decrypt credentials: %s", e)
            raise ValueError("Invalid or corrupted credential data") from e

    def _sanitize_credentials(
        self, credentials: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Remove sensitive information for logging."""
        sensitive_keys = {
            'password', 'passwd', 'pwd', 'secret', 'token', 'key', 'pat', 'api_key'
        }
        # Check if key contains any sensitive substrings
        return {
            k: '***REDACTED***' if any(sens in k.lower() for sens in sensitive_keys) and v else v
            for k, v in credentials.items()
        }


class SQLInjectionValidator:
    """Validates SQL queries to prevent injection attacks."""

    # Dangerous SQL patterns that should be blocked
    DANGEROUS_PATTERNS = [
        # Multiple statements with DDL/DML
        r';.*(?:DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE)',
        # Classic injection patterns (but NOT blocking UNION/UNION ALL)
        r'OR\s+1\s*=\s*1',  # Always true conditions
        r'AND\s+1\s*=\s*1',  # Always true conditions
        r"OR\s+'[^']*'\s*=\s*'[^']*'",  # String-based always true
        # SQL comment injection patterns
        r"'[^']*'\s*--",  # String literal followed by SQL comment (injection pattern)
        r'/\*.*\*/',  # Block comments (can be used to obfuscate or inject)
        # Multiple semicolons or semicolon with dangerous operations
        r';\s*;',  # Multiple semicolons
        r';\s*(?:DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE)',  # Semicolon followed by DDL/DML
        # SQL Server command execution
        r'xp_cmdshell|sp_executesql|exec\s*\(',
        # MySQL file operations
        r'load_file|into\s+outfile|into\s+dumpfile',
        # PostgreSQL file operations
        r'pg_read_file|copy.*from|copy.*to',
        # System tables that shouldn't be accessed directly in normal queries
        r'information_schema\.(?:user_privileges|schema_privileges|table_privileges)',
        r'pg_catalog\.pg_authid|pg_catalog\.pg_user_mapping',
        # Dangerous UNION queries attempting to extract sensitive data
        r'UNION\s+(?:ALL\s+)?SELECT\s+.*(?:password|pwd|secret|token|key|admin)',
    ]

    # Safe SQL patterns that are allowed
    # Optional comment prefix: allows -- comments and /* */ comments at start of query
    # Note: After _clean_query(), newlines become spaces, so -- comment ends at next SQL keyword
    # Pattern matches: optional (-- followed by non-SQL text, or /* ... */) followed by whitespace
    _OPT_COMMENT = r'^(?:--[^;]*?\s+)?'

    SAFE_PATTERNS = [
        # Basic SELECT queries with JOINs (with optional leading comments)
        _OPT_COMMENT + r'SELECT\s+.+\s+FROM\s+.+$',
        # CTEs (WITH clauses) - now more permissive for complex queries
        _OPT_COMMENT + r'WITH\s+.+\s+SELECT\s+.+$',
        # Metadata queries
        _OPT_COMMENT + r'DESCRIBE\s+[\w\."]+$',
        _OPT_COMMENT + r'DESC\s+[\w\."]+$',
        _OPT_COMMENT + r'SHOW\s+(?:TABLES|COLUMNS|DATABASES|SCHEMAS)(?:\s+FROM\s+[\w\."]+)?$',
        _OPT_COMMENT + r'EXPLAIN\s+.+$',
        # UNION queries (both UNION and UNION ALL) - Note: Will still be blocked if dangerous patterns match
        r'.+\s+UNION\s+(?:ALL\s+)?SELECT\s+.+',
    ]

    def __init__(self) -> None:
        self.compiled_dangerous = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.DANGEROUS_PATTERNS
        ]
        self.compiled_safe = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.SAFE_PATTERNS
        ]

    def validate_query(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query for security issues."""
        if not sql_query or not sql_query.strip():
            return {
                "is_safe": False,
                "risk_level": SecurityLevel.HIGH.value,
                "issues": ["Empty query"],
                "sanitized_query": ""
            }

        # Clean up query for analysis
        cleaned_query = self._clean_query(sql_query)
        issues = []
        risk_level = SecurityLevel.LOW

        # Check for dangerous patterns
        for pattern in self.compiled_dangerous:
            if pattern.search(cleaned_query):
                issues.append(
                    f"Potentially dangerous SQL pattern detected: "
                    f"{pattern.pattern}"
                )
                risk_level = SecurityLevel.CRITICAL

        # Check for multiple statements
        statements = [s.strip() for s in cleaned_query.split(';') if s.strip()]
        if len(statements) > 1:
            issues.append("Multiple SQL statements not allowed")
            risk_level = SecurityLevel.CRITICAL

        # Validate against safe patterns (if no dangerous patterns found)
        is_safe = len(issues) == 0
        if is_safe and statements:
            statement = statements[0]
            matches_safe_pattern = any(
                pattern.match(statement) for pattern in self.compiled_safe
            )
            if not matches_safe_pattern:
                issues.append("Query does not match approved safe patterns")
                risk_level = SecurityLevel.MEDIUM
                is_safe = False

        return {
            "is_safe": is_safe,
            "risk_level": risk_level.value,
            "issues": issues,
            "sanitized_query": self._sanitize_query_for_logging(cleaned_query)
        }

    def _clean_query(self, sql_query: str) -> str:
        """Clean and normalize SQL query for analysis."""
        # Remove leading/trailing whitespace
        cleaned = sql_query.strip()

        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned

    def _sanitize_query_for_logging(
        self, sql_query: str, max_length: int = 200
    ) -> str:
        """Sanitize SQL query for safe logging."""
        # Remove potential sensitive data patterns
        sanitized = re.sub(r"'[^']*'", "'***'", sql_query)  # String literals
        # Quoted identifiers with potential data
        sanitized = re.sub(r'"[^"]*"', '"***"', sanitized)

        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."

        return sanitized


class IdentifierValidator:
    """Validates database identifiers to prevent injection."""

    # Valid identifier pattern (letters, numbers, underscores only - no hyphens)
    VALID_IDENTIFIER = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    # Valid schema.table pattern (no hyphens)
    VALID_QUALIFIED_IDENTIFIER = re.compile(
        r'^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$'
    )

    @classmethod
    def validate_identifier(cls, identifier: str) -> bool:
        """Validate a single database identifier."""
        if not identifier:
            return False

        # Check length (reasonable limit)
        if len(identifier) > 128:
            return False

        return bool(cls.VALID_IDENTIFIER.match(identifier))

    @classmethod
    def validate_qualified_identifier(cls, identifier: str) -> bool:
        """Validate schema.table or database.schema.table identifier."""
        if not identifier:
            return False

        parts = identifier.split('.')
        if len(parts) > 3:  # database.schema.table max
            return False

        return all(cls.validate_identifier(part) for part in parts)

    @classmethod
    def sanitize_identifier(cls, identifier: str) -> str:
        """Sanitize identifier for safe usage."""
        if not identifier:
            return ""

        # Remove invalid characters (including hyphens)
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', identifier)

        # Ensure starts with letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = '_' + sanitized

        # Limit length
        return sanitized[:128]


def audit_log_security_event(
    event_type: str,
    details: Dict[str, Any],
    risk_level: SecurityLevel = SecurityLevel.MEDIUM
) -> None:
    """Log security-related events for auditing."""
    # Sanitize details before logging
    safe_details = {}
    sensitive_keys = {
        'password', 'passwd', 'pwd', 'secret', 'token', 'key', 'pat', 'api_key'
    }

    for key, value in details.items():
        # Check if key contains any sensitive substring
        if any(sens in key.lower() for sens in sensitive_keys):
            safe_details[key] = '***REDACTED***'
        else:
            safe_details[key] = value

    # Use f-string formatting to actually substitute values
    logger.warning(
        f"SECURITY_AUDIT: {event_type} | Risk: {risk_level.value} | Details: {safe_details}"
    )


# Global instances
sql_validator = SQLInjectionValidator()
identifier_validator = IdentifierValidator()
