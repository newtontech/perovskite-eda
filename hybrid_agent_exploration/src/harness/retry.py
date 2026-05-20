"""retry.py — Auto-retry for transient failures."""

import time
import traceback
from typing import Any, Callable, Tuple


class RetryPolicy:
    """Execute a function with exponential-backoff retries on transient errors.

    Retries are triggered for ``MemoryError``, timeout-related exceptions,
    and any exception whose traceback originates in RDKit.
    """

    def __init__(
        self,
        max_retries: int = 1,
        retryable_exceptions: Tuple[type, ...] | None = None,
    ):
        self.max_retries = max_retries
        self.retryable_exceptions = self._build_exceptions(retryable_exceptions)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_exceptions(user_exceptions: Tuple[type, ...] | None) -> Tuple[type, ...]:
        base = (MemoryError, TimeoutError)
        try:
            from concurrent.futures import TimeoutError as FutureTimeout

            base = base + (FutureTimeout,)
        except ImportError:
            pass
        if user_exceptions is not None:
            base = base + user_exceptions
        return base

    @staticmethod
    def _originates_from_rdkit(exc: BaseException) -> bool:
        """Walk the traceback and return *True* if any frame is inside *rdkit*."""
        mod = getattr(type(exc), "__module__", "")
        if mod and "rdkit" in mod.lower():
            return True
        tb = exc.__traceback__
        while tb is not None:
            frame_mod = tb.tb_frame.f_globals.get("__name__", "")
            if "rdkit" in frame_mod.lower():
                return True
            tb = tb.tb_next
        return False

    def _is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, self.retryable_exceptions):
            return True
        return self._originates_from_rdkit(exc)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Call *func* and retry on transient failures."""
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc):
                    raise
                if attempt < self.max_retries:
                    backoff = 2 ** attempt
                    time.sleep(backoff)
        # Exhausted all retries – re-raise the last exception
        assert last_exc is not None
        raise last_exc
