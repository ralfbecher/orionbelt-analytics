# Integrating Database Ontology MCP Server with Claude Desktop

This guide explains how to integrate your Database Ontology MCP Server with Claude Desktop.

## Prerequisites

1. **Claude Desktop App**: Make sure you have Claude Desktop installed
2. **UV Package Manager**: Ensure `uv` is installed at `/Users/ralfbecher/.local/bin/uv`
3. **Project Dependencies**: Make sure all dependencies are installed

## Installation Steps

### 1. Install Dependencies with UV

First, install the project dependencies using `uv`:

```bash
cd /Users/ralfbecher/database-ontology-mcp
/Users/ralfbecher/.local/bin/uv sync
```

### 2. Set Up Environment Variables

Copy the environment template file to create your local configuration:

```bash
# Copy the template to create your local .env file
cp .env.template .env
```

Edit the `.env` file with your database credentials:

```env
# Database connection settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=mydb
POSTGRES_USERNAME=user
POSTGRES_PASSWORD=password

# Snowflake settings (if using Snowflake)
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_USERNAME=user
SNOWFLAKE_PASSWORD=password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=MYDB
SNOWFLAKE_SCHEMA=PUBLIC

# Additional configuration as needed

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
```

### 3. Test the Server

Test that the server works correctly:

```bash
/Users/ralfbecher/.local/bin/uv run python run_server.py
```

You should see logging output indicating the server has started and listing all available tools.

### 4. Configure Claude Desktop

#### Option A: Using the provided configuration file

1. Copy the contents of `claude_mcp_config.json`
2. Open Claude Desktop settings
3. Go to the "Developer" or "MCP Servers" section
4. Add the configuration

#### Option B: Manual configuration

Add this configuration to your Claude Desktop MCP servers settings:

```json
{
  "mcpServers": {
    "database-ontology": {
      "command": "/Users/ralfbecher/.local/bin/uv",
      "args": [
        "run",
        "--project",
        "/Users/ralfbecher/database-ontology-mcp",
        "python",
        "run_server.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/ralfbecher/database-ontology-mcp"
      }
    }
  }
}
```

## Available Tools

The MCP server provides these tools:

- **connect_database**: Connect to PostgreSQL or Snowflake databases
- **list_schemas**: List available database schemas
- **analyze_schema**: Analyze tables and columns in a specific schema
- **generate_ontology**: Generate RDF/OWL ontology from database schema
- **sample_table_data**: Sample data from specific tables
- **get_table_relationships**: Get foreign key relationships between tables
- **get_server_info**: Get server information and capabilities

## Features

- **Enhanced Error Handling**: Standardized error responses with detailed error types
- **Input Validation**: Comprehensive parameter validation for all tools
- **Connection Management**: Improved database connection handling with proper cleanup
- **LLM Integration**: Optional ontology enrichment using LLM capabilities via MCP
- **Logging**: Configurable logging levels via environment variables
- **Security**: SQL injection protection and input sanitization

## Usage Examples

Once connected to Claude Desktop, you can use natural language to interact with the tools:

- "Connect to my PostgreSQL database on localhost"
- "Show me all schemas in the database"
- "Analyze the 'public' schema structure"
- "Generate an ontology from the user_management schema"
- "Sample some data from the users table"
- "Show me the relationships between tables"

## Troubleshooting

1. **Connection Issues**: Check your database credentials in the `.env` file
2. **Server Not Starting**: Verify all dependencies are installed with `uv sync`
3. **Permission Errors**: Ensure the UV path is correct and executable
4. **Logging**: Set `LOG_LEVEL=DEBUG` for detailed troubleshooting information

## Security Notes

- Never commit your `.env` file to version control
- Use environment-specific credentials
- Consider using connection pooling for production deployments
- The server includes SQL injection protection, but always validate your database permissions
