"""Langfuse tracing client (v4).

Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from the
environment. If keys are absent the client is disabled and every call
below is a cheap no-op, so instrumentation can stay unconditional.
"""

from dotenv import load_dotenv
from langfuse import get_client

# `uv run uvicorn` does not export .env into os.environ (pydantic loads it only
# for Settings), but the Langfuse SDK reads LANGFUSE_* from os.environ. Load it
# here for native runs; a no-op in Docker where env comes from the container.
load_dotenv()

langfuse = get_client()
