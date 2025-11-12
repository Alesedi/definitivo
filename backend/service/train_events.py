import asyncio
import json
from typing import Dict

# Simple in-memory pub/sub for training events per source
_queues: Dict[str, asyncio.Queue] = {}


def get_queue_for_source(source: str):
    src = (source or 'tmdb').lower()
    if src not in _queues:
        _queues[src] = asyncio.Queue()
    return _queues[src]


async def publish_event(source: str, event: dict):
    q = get_queue_for_source(source)
    await q.put(event)


async def event_generator(source: str):
    """Async generator that yields SSE formatted strings for a given source."""
    q = get_queue_for_source(source)
    while True:
        try:
            event = await q.get()
            data = json.dumps(event)
            yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.5)
