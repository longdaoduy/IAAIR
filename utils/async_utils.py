"""
Async utilities for offloading blocking (synchronous) work
from the FastAPI/asyncio event loop.

Usage:
    from utils.async_utils import run_blocking

    # In an async endpoint or method:
    result = await run_blocking(some_sync_function, arg1, arg2)
"""

import asyncio
from functools import partial
from typing import TypeVar, Callable, Any
from concurrent.futures import ThreadPoolExecutor

T = TypeVar("T")

# Shared thread pool for CPU / IO-bound blocking work.
# Size = min(32, cpu_count + 4) by default in Python 3.8+,
# but we cap it explicitly so heavy ML inference doesn't starve other tasks.
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="iaair-blocking")


async def run_blocking(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous (blocking) function in a thread pool
    without blocking the asyncio event loop.

    Args:
        func: The synchronous function to call.
        *args: Positional arguments forwarded to *func*.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        The return value of *func(*args, **kwargs)*.

    Example::

        embedding = await run_blocking(
            scibert_client.generate_text_embedding, query_text
        )
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(_executor, func, *args)
