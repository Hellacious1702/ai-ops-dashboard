"""
LLM Instrumentation middleware for LiteLLM.
Auto-instruments all LLM calls to capture latency, tokens, cost, errors, and model info.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any

import litellm

from ai_ops.core.storage.database import get_db_session
from ai_ops.shared.config import get_settings
from ai_ops.shared.models import CallStatus, LLMCall, ModelProvider, estimate_cost
from ai_ops.shared.utils import Event, EventType, event_bus, generate_id, now_utc


class LLMInstrumentor:
    """
    Middleware wrapper for LiteLLM that automatically instruments all LLM calls.
    
    WHY: We need observability into every LLM call without modifying application code.
    This middleware wraps LiteLLM's completion/acompletion functions to capture:
    - Latency (wall-clock time)
    - Token counts (prompt, completion, total)
    - Estimated cost (using model pricing table)
    - Errors and their types
    - Model and provider identification
    
    The middleware uses LiteLLM's callback system for non-intrusive instrumentation.
    """

    def __init__(self) -> None:
        self.settings = get_settings().instrumentation
        self._enabled = self.settings.enabled
        self._original_completion = None
        self._original_acompletion = None
        self._call_buffer: list[LLMCall] = []
        self._buffer_lock = None
        self._flush_task: Any | None = None

    def enable(self) -> None:
        """Enable instrumentation by patching LiteLLM."""
        if self._enabled:
            return

        self._enabled = True
        self._patch_litellm()
        self._start_flush_task()

    def disable(self) -> None:
        """Disable instrumentation and restore original functions."""
        if not self._enabled:
            return

        self._enabled = False
        self._unpatch_litellm()
        self._stop_flush_task()

    def _patch_litellm(self) -> None:
        """Patch LiteLLM's completion functions with instrumentation."""
        # Store originals
        self._original_completion = litellm.completion
        self._original_acompletion = litellm.acompletion

        # Create instrumented wrappers
        @wraps(self._original_completion)
        def instrumented_completion(*args: Any, **kwargs: Any) -> Any:
            return self._instrumented_call(self._original_completion, *args, **kwargs)

        @wraps(self._original_acompletion)
        async def instrumented_acompletion(*args: Any, **kwargs: Any) -> Any:
            return await self._instrumented_acall(self._original_acompletion, *args, **kwargs)

        # Apply patches
        litellm.completion = instrumented_completion
        litellm.acompletion = instrumented_acompletion

        # Also patch the module-level functions
        import litellm.main
        litellm.main.completion = instrumented_completion
        litellm.main.acompletion = instrumented_acompletion

    def _unpatch_litellm(self) -> None:
        """Restore original LiteLLM functions."""
        if self._original_completion:
            litellm.completion = self._original_completion
            litellm.acompletion = self._original_acompletion
            import litellm.main
            litellm.main.completion = self._original_completion
            litellm.main.acompletion = self._original_acompletion

    def _instrumented_call(self, original_func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Synchronous instrumented call wrapper."""
        start_time = time.perf_counter()
        call_id = generate_id("call_")

        # Extract model and provider from kwargs
        model = kwargs.get("model", "unknown")
        provider = self._extract_provider(model, kwargs)

        # Hash prompt for deduplication (if enabled)
        prompt_hash = None
        if not self.settings.capture_prompt:
            prompt_hash = self._hash_prompt(kwargs.get("messages", []))

        try:
            # Execute original call
            response = original_func(*args, **kwargs)

            # Extract metrics from response
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Parse response for token usage
            prompt_tokens, completion_tokens, total_tokens = self._extract_tokens(response)

            # Calculate cost
            cost_usd = estimate_cost(provider, model, prompt_tokens, completion_tokens)

            # Create LLMCall record
            call = LLMCall(
                id=call_id,
                timestamp=now_utc(),
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                status=CallStatus.SUCCESS,
                prompt_hash=prompt_hash,
                metadata={"response_id": getattr(response, "id", None)},
            )

            # Buffer for async write
            self._buffer_call(call)

            # Emit event
            event_bus.emit_sync(Event(
                type=EventType.LLM_CALL_END,
                payload=call.model_dump(),
            ))

            return response

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            status = self._classify_error(e)

            call = LLMCall(
                id=call_id,
                timestamp=now_utc(),
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                status=status,
                error_message=str(e),
                prompt_hash=prompt_hash,
            )

            self._buffer_call(call)

            event_bus.emit_sync(Event(
                type=EventType.LLM_CALL_ERROR,
                payload=call.model_dump(),
            ))

            raise

    async def _instrumented_acall(self, original_func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Asynchronous instrumented call wrapper."""
        start_time = time.perf_counter()
        call_id = generate_id("call_")

        model = kwargs.get("model", "unknown")
        provider = self._extract_provider(model, kwargs)

        prompt_hash = None
        if not self.settings.capture_prompt:
            prompt_hash = self._hash_prompt(kwargs.get("messages", []))

        try:
            response = await original_func(*args, **kwargs)

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            prompt_tokens, completion_tokens, total_tokens = self._extract_tokens(response)
            cost_usd = estimate_cost(provider, model, prompt_tokens, completion_tokens)

            call = LLMCall(
                id=call_id,
                timestamp=now_utc(),
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                status=CallStatus.SUCCESS,
                prompt_hash=prompt_hash,
                metadata={"response_id": getattr(response, "id", None)},
            )

            self._buffer_call(call)

            await event_bus.emit(Event(
                type=EventType.LLM_CALL_END,
                payload=call.model_dump(),
            ))

            return response

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            status = self._classify_error(e)

            call = LLMCall(
                id=call_id,
                timestamp=now_utc(),
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                status=status,
                error_message=str(e),
                prompt_hash=prompt_hash,
            )

            self._buffer_call(call)

            await event_bus.emit(Event(
                type=EventType.LLM_CALL_ERROR,
                payload=call.model_dump(),
            ))

            raise

    def _extract_provider(self, model: str, kwargs: dict) -> ModelProvider:
        """Extract provider from model name or kwargs."""
        # Check explicit provider in kwargs
        if "custom_llm_provider" in kwargs:
            provider_str = kwargs["custom_llm_provider"].lower()
            try:
                return ModelProvider(provider_str)
            except ValueError:
                pass

        # Infer from model name
        model_lower = model.lower()
        if model_lower.startswith("gpt-") or model_lower.startswith("o1-") or "openai" in model_lower:
            return ModelProvider.OPENAI
        elif model_lower.startswith("claude-") or "anthropic" in model_lower:
            return ModelProvider.ANTHROPIC
        elif model_lower.startswith("azure/") or "azure" in model_lower:
            return ModelProvider.AZURE
        elif model_lower.startswith("cohere/") or "command" in model_lower:
            return ModelProvider.COHERE
        elif model_lower.startswith("huggingface/") or "hf/" in model_lower:
            return ModelProvider.HUGGINGFACE
        elif model_lower.startswith("ollama/") or "ollama" in model_lower:
            return ModelProvider.OLLAMA
        elif model_lower.startswith("vertex/") or "gemini" in model_lower:
            return ModelProvider.VERTEX
        elif model_lower.startswith("bedrock/") or "bedrock" in model_lower:
            return ModelProvider.BEDROCK

        return ModelProvider.OTHER

    def _extract_tokens(self, response: Any) -> tuple[int, int, int]:
        """Extract token counts from LiteLLM response."""
        try:
            # LiteLLM response has usage attribute
            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
                return prompt_tokens, completion_tokens, total_tokens
        except Exception:
            pass

        # Fallback: try to parse from response dict
        try:
            if hasattr(response, "model_dump"):
                response_dict = response.model_dump()
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {}

            usage = response_dict.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            return prompt_tokens, completion_tokens, total_tokens
        except Exception:
            pass

        return 0, 0, 0

    def _hash_prompt(self, messages: list[dict]) -> str:
        """Create a hash of the prompt for deduplication."""
        try:
            content = str(messages)
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception:
            return ""

    def _classify_error(self, error: Exception) -> CallStatus:
        """Classify error type for metrics."""
        error_str = str(error).lower()
        if "timeout" in error_str:
            return CallStatus.TIMEOUT
        elif "rate limit" in error_str or "429" in error_str:
            return CallStatus.RATE_LIMITED
        elif "cancel" in error_str:
            return CallStatus.CANCELLED
        return CallStatus.ERROR

    def _buffer_call(self, call: LLMCall) -> None:
        """Buffer call for async write."""
        self._call_buffer.append(call)

        # Flush if buffer is full
        if len(self._call_buffer) >= self.settings.async_buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffered calls to database."""
        if not self._call_buffer:
            return

        calls = self._call_buffer
        self._call_buffer = []

        # Write to database asynchronously
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._write_calls(calls))
            else:
                loop.run_until_complete(self._write_calls(calls))
        except RuntimeError:
            # No event loop, schedule for later
            pass

    async def _write_calls(self, calls: list[LLMCall]) -> None:
        """Write calls to database."""
        async with get_db_session() as session:
            for call in calls:
                model = call.to_model()
                session.add(model)
            await session.commit()

    def _start_flush_task(self) -> None:
        """Start periodic buffer flush task."""
        import asyncio

        async def flush_loop():
            while self._enabled:
                await asyncio.sleep(self.settings.flush_interval_seconds)
                self._flush_buffer()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._flush_task = loop.create_task(flush_loop())
        except RuntimeError:
            pass

    def _stop_flush_task(self) -> None:
        """Stop periodic buffer flush task."""
        if self._flush_task:
            self._flush_task.cancel()
            self._flush_task = None

        # Flush remaining
        self._flush_buffer()

    @asynccontextmanager
    async def track_call(self, model: str, provider: ModelProvider, **kwargs: Any):
        """
        Context manager for manually tracking LLM calls.
        
        Usage:
            async with instrumentor.track_call("gpt-4", ModelProvider.OPENAI) as call:
                response = await my_llm_call()
                call.prompt_tokens = 100
                call.completion_tokens = 50
        """
        call_id = generate_id("call_")
        start_time = time.perf_counter()

        call = LLMCall(
            id=call_id,
            timestamp=now_utc(),
            provider=provider,
            model=model,
        )

        try:
            yield call
        finally:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            call.latency_ms = latency_ms
            call.total_tokens = call.prompt_tokens + call.completion_tokens
            call.cost_usd = estimate_cost(provider, model, call.prompt_tokens, call.completion_tokens)

            self._buffer_call(call)

            await event_bus.emit(Event(
                type=EventType.LLM_CALL_END,
                payload=call.model_dump(),
            ))


# Global instrumentor instance
_instrumentor: LLMInstrumentor | None = None


def get_instrumentor() -> LLMInstrumentor:
    """Get or create the global instrumentor."""
    global _instrumentor
    if _instrumentor is None:
        _instrumentor = LLMInstrumentor()
    return _instrumentor


def init_instrumentation() -> LLMInstrumentor:
    """Initialize and enable instrumentation."""
    instrumentor = get_instrumentor()
    instrumentor.enable()
    return instrumentor


def shutdown_instrumentation() -> None:
    """Shutdown instrumentation."""
    global _instrumentor
    if _instrumentor:
        _instrumentor.disable()
        _instrumentor = None
