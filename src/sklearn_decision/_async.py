from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

__all__ = ["run_coro"]


def run_coro(coro):
    """asyncio.run that also works inside Jupyter's running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()
