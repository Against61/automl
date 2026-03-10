from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import ResponseError


def _encode_payload(payload: dict[str, Any]) -> dict[str, str]:
    return {"payload": json.dumps(payload, ensure_ascii=True)}


def _decode_payload(fields: dict[str, str | bytes]) -> dict[str, Any]:
    raw = fields.get("payload")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


class RedisEventBus:
    def __init__(
        self,
        redis_url: str,
        stream_tasks: str,
        stream_control: str,
        stream_internal: str,
        stream_results: str,
        consumer_group: str,
        consumer_name: str,
    ) -> None:
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.stream_tasks = stream_tasks
        self.stream_control = stream_control
        self.stream_internal = stream_internal
        self.stream_results = stream_results
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name

    async def initialize(self) -> None:
        await self._ensure_group(self.stream_tasks)
        await self._ensure_group(self.stream_control)

    async def _ensure_group(self, stream: str) -> None:
        try:
            await self.redis.xgroup_create(name=stream, groupname=self.consumer_group, id="$", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def close(self) -> None:
        await self.redis.aclose()

    async def publish_internal(self, event_type: str, payload: dict[str, Any]) -> str:
        message = {"event_id": str(uuid4()), "event_type": event_type, **payload}
        return await self.redis.xadd(self.stream_internal, _encode_payload(message))

    async def publish_task(self, payload: dict[str, Any]) -> str:
        return await self.redis.xadd(self.stream_tasks, _encode_payload(payload))

    async def publish_result(self, payload: dict[str, Any]) -> str:
        return await self.redis.xadd(self.stream_results, _encode_payload(payload))

    async def publish_control(self, payload: dict[str, Any]) -> str:
        return await self.redis.xadd(self.stream_control, _encode_payload(payload))

    async def _read_stream(
        self,
        stream: str,
        block_ms: int = 5000,
        count: int = 10,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        while True:
            messages = await self.redis.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={stream: ">"},
                count=count,
                block=block_ms,
            )
            if not messages:
                yield "", {}
                continue
            for _, entries in messages:
                for message_id, fields in entries:
                    yield message_id, _decode_payload(fields)

    async def iter_task_events(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        async for message_id, payload in self._read_stream(self.stream_tasks):
            yield message_id, payload

    async def iter_control_events(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        async for message_id, payload in self._read_stream(self.stream_control):
            yield message_id, payload

    async def ack_task(self, message_id: str) -> None:
        if message_id:
            await self.redis.xack(self.stream_tasks, self.consumer_group, message_id)

    async def ack_control(self, message_id: str) -> None:
        if message_id:
            await self.redis.xack(self.stream_control, self.consumer_group, message_id)


class InMemoryEventBus:
    def __init__(self) -> None:
        self.task_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self.control_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self.internal: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self._counter = 0

    async def initialize(self) -> None:
        return

    async def close(self) -> None:
        return

    async def publish_task(self, payload: dict[str, Any]) -> str:
        self._counter += 1
        message_id = f"{self._counter}-0"
        await self.task_queue.put((message_id, payload))
        return message_id

    async def publish_control(self, payload: dict[str, Any]) -> str:
        self._counter += 1
        message_id = f"{self._counter}-0"
        await self.control_queue.put((message_id, payload))
        return message_id

    async def publish_internal(self, event_type: str, payload: dict[str, Any]) -> str:
        event = {"event_id": str(uuid4()), "event_type": event_type, **payload}
        self.internal.append(event)
        self._counter += 1
        return f"{self._counter}-0"

    async def publish_result(self, payload: dict[str, Any]) -> str:
        self.results.append(payload)
        self._counter += 1
        return f"{self._counter}-0"

    async def iter_task_events(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        while True:
            yield await self.task_queue.get()

    async def iter_control_events(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        while True:
            yield await self.control_queue.get()

    async def ack_task(self, message_id: str) -> None:
        return

    async def ack_control(self, message_id: str) -> None:
        return
