# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

We take the security of OrionBelt Analytics seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

**Please DO NOT open a public issue.** Instead:

1. **Email**: Send details to ralf.becher@web.de
2. **Subject**: Include "[SECURITY]" in the subject line
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Suggested fix (if you have one)
   - Your contact information

### What to Expect

- **Initial Response**: Within 48 hours
- **Status Update**: Within 5 business days
- **Fix Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 7-14 days
  - Medium: 14-30 days
  - Low: 30-90 days

### Security Considerations

This project handles sensitive database credentials and executes SQL queries. Security concerns include:

- **SQL Injection**: We implement comprehensive validation and parameterized queries
- **Credential Management**: Supports encryption with master password (AES-128-CBC with HMAC)
- **Access Control**: Database permissions should follow principle of least privilege
- **Data Exposure**: Query results may contain sensitive information

### Best Practices for Users

When deploying OrionBelt Analytics:

1. **Credentials**:
   - Use strong master password (MCP_MASTER_PASSWORD)
   - Restrict .env file permissions: `chmod 600 .env`
   - Never commit .env files to version control
   - Rotate credentials regularly

2. **Database Access**:
   - Use read-only database accounts when possible
   - Implement row-level security in your database
   - Limit schema access to what's necessary

3. **Network Security**:
   - Run behind a firewall
   - Use HTTPS/TLS for database connections
   - Restrict MCP server access to trusted clients only

4. **Monitoring**:
   - Enable detailed logging (LOG_LEVEL=INFO or DEBUG)
   - Monitor for unusual query patterns
   - Review access logs regularly

## Security Updates

Security patches will be announced via:
- GitHub Security Advisories
- Release notes
- Project README

## Acknowledgments

We appreciate the security research community's efforts to responsibly disclose vulnerabilities and will acknowledge contributors (with permission) in our security advisories.

## Contact

For security concerns: ralf.becher@web.de

For general questions: Use GitHub Issues

---

*Last updated: October 2025*
