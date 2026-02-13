"""Dremio REST API client for database operations."""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class DremioAuthError(Exception):
    """Authentication error with Dremio."""
    pass


class DremioQueryError(Exception):
    """Query execution error in Dremio."""
    pass


class DremioClient:
    """Dremio REST API client."""
    
    def __init__(self, uri: str, username: str = None, password: str = None, token: str = None, pat: str = None):
        self.uri = uri.rstrip('/')
        self.username = username
        self.password = password
        self.token = token
        self.pat = pat  # Personal Access Token (preferred)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        if self.pat:
            # Use PAT directly as token (following official dremio-mcp approach)
            self.token = self.pat
            logger.info("Using Personal Access Token (PAT) for authentication")
        elif not self.token and self.username and self.password:
            # Fall back to username/password authentication
            await self._authenticate()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _authenticate(self):
        """Authenticate with username/password to get a token (following official Dremio examples)."""
        auth_data = {
            "userName": self.username,  # Official Dremio API uses 'userName' not 'username'
            "password": self.password
        }
        
        try:
            async with self.session.post(
                f"{self.uri}/apiv2/login",
                json=auth_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.token = data.get("token")
                    if not self.token:
                        raise DremioAuthError("No token received from authentication")
                    logger.info("Successfully authenticated with Dremio")
                else:
                    error_text = await response.text()
                    logger.error(f"Authentication failed: {response.status} - {error_text}")
                    raise DremioAuthError(f"Authentication failed: {response.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Network error during authentication: {e}")
            raise DremioAuthError(f"Network error: {e}")
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to Dremio API."""
        if not self.token:
            raise DremioAuthError("No authentication token available")
            
        headers = {
            "Authorization": f"Bearer {self.token}" if self.pat else f"_dremio{self.token}",
            "Content-Type": "application/json"
        }
        if 'headers' in kwargs:
            headers.update(kwargs.pop('headers'))
            
        url = f"{self.uri}{endpoint}"
        logger.debug(f"Making {method} request to {url}")
        
        try:
            async with self.session.request(method, url, headers=headers, **kwargs) as response:
                if response.status == 401:
                    raise DremioAuthError("Authentication token expired or invalid")
                
                if response.content_type == 'application/json':
                    data = await response.json()
                else:
                    data = {"text": await response.text()}
                
                if response.status >= 400:
                    error_msg = data.get('errorMessage', f"HTTP {response.status}")
                    raise DremioQueryError(f"API request failed: {error_msg}")
                
                return data
        except aiohttp.ClientError as e:
            logger.error(f"Network error during API request: {e}")
            raise DremioQueryError(f"Network error: {e}")
    
    async def execute_query(self, sql: str, limit: int = 1000) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        try:
            # Log the SQL query being executed
            logger.info(f"🔍 DREMIO SQL QUERY: {sql}")
            
            # Submit query
            query_data = {"sql": sql}
            job = await self._make_request("POST", "/api/v3/sql", json=query_data)
            job_id = job.get("id")
            
            if not job_id:
                raise DremioQueryError("No job ID returned from query submission")
            
            logger.info(f"Submitted query with job ID: {job_id}")
            
            # Poll for completion
            while True:
                job_status = await self._make_request("GET", f"/api/v3/job/{job_id}")
                state = job_status.get("jobState")
                
                if state in ["COMPLETED"]:
                    break
                elif state in ["FAILED", "CANCELED"]:
                    error = job_status.get("errorMessage", "Query failed")
                    raise DremioQueryError(f"Query {state.lower()}: {error}")
                elif state in ["RUNNING", "STARTING", "PLANNING", "PENDING", "QUEUED", "METADATA_RETRIEVAL"]:
                    await asyncio.sleep(0.5)  # Poll every 500ms
                else:
                    logger.warning(f"Unknown job state: {state}")
                    await asyncio.sleep(0.5)
            
            # Get results
            row_count = job_status.get("rowCount", 0)
            if row_count == 0:
                return {
                    "success": True,
                    "data": [],
                    "row_count": 0,
                    "columns": []
                }
            
            # Fetch results with limit
            actual_limit = min(limit, row_count)
            results = await self._make_request(
                "GET", 
                f"/api/v3/job/{job_id}/results",
                params={"offset": 0, "limit": actual_limit}
            )
            
            rows = results.get("rows", [])
            schema = results.get("schema", [])
            columns = [col.get("name") for col in schema]
            
            return {
                "success": True,
                "data": rows,
                "row_count": len(rows),
                "total_rows": row_count,
                "columns": columns,
                "job_id": job_id
            }
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def get_catalogs(self) -> List[Dict[str, Any]]:
        """Get list of catalogs (schemas)."""
        try:
            catalogs = await self._make_request("GET", "/api/v3/catalog")
            data = catalogs.get("data", [])
            # Process catalog data to extract proper structure
            result = []
            for item in data:
                # Handle path which might be a list
                path = item.get("path", [])
                if isinstance(path, list) and path:
                    name = path[-1]  # Get the last element as the name
                else:
                    name = path if isinstance(path, str) else ""
                
                result.append({
                    "name": name,
                    "path": path,
                    "type": item.get("type", ""),
                    "id": item.get("id", ""),
                    "tag": item.get("tag", "")
                })
            return result
        except Exception as e:
            logger.error(f"Failed to get catalogs: {e}")
            return []
    
    async def get_catalog_info(self, path: List[str]) -> Dict[str, Any]:
        """Get information about a catalog item."""
        try:
            # For nested paths, join with slashes (Dremio REST API requirement)
            if isinstance(path, list):
                catalog_path = "/".join(path)
            else:
                # If already a string, convert dots to slashes if present
                catalog_path = str(path).replace(".", "/")
            info = await self._make_request("GET", f"/api/v3/catalog/by-path/{catalog_path}")
            return info
        except Exception as e:
            logger.error(f"Failed to get catalog info for {path}: {e}")
            return {}
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Dremio."""
        try:
            # Simple query to test connection
            result = await self.execute_query("SELECT 1 as test_column")
            if result.get("success"):
                return {
                    "success": True,
                    "message": "Connection test successful",
                    "server_info": "Dremio REST API"
                }
            else:
                return {
                    "success": False,
                    "error": "Connection test query failed",
                    "details": result.get("error")
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Connection test failed: {str(e)}",
                "error_type": type(e).__name__
            }


async def create_dremio_client(host: str = None, port: int = 9047, username: str = None, 
                              password: str = None, token: str = None, pat: str = None,
                              uri: str = None, ssl: bool = True) -> DremioClient:
    """Create and authenticate Dremio client.
    
    Supports both new PAT-based authentication and legacy username/password.
    
    Args:
        uri: Full Dremio API URI (preferred, overrides host/port/ssl)
        pat: Personal Access Token (preferred authentication method)
        host: Dremio host (legacy)
        port: Dremio port (legacy, default 9047)
        username: Username (legacy)
        password: Password (legacy)
        token: Existing token (legacy)
        ssl: Enable SSL (legacy, default True)
    """
    if uri:
        # Use provided URI directly (official dremio-mcp approach)
        final_uri = uri
    elif host:
        # Build URI from host/port/ssl (legacy approach)
        protocol = "https" if ssl else "http"
        final_uri = f"{protocol}://{host}:{port}"
    else:
        raise ValueError("Either 'uri' or 'host' must be provided")
    
    client = DremioClient(uri=final_uri, username=username, password=password, token=token, pat=pat)
    return client