"""sandbox.py — Execution environment isolation for worker agents."""

import multiprocessing as mp
import os
import sys
import time
from typing import Any, Callable


class Sandbox:
    """Run functions in isolated processes with timeout and memory guards.

    Uses ``multiprocessing.get_context("spawn")`` for full isolation.
    *func* must be a top-level, picklable function.
    """

    def __init__(self, timeout_sec: int = 300, max_memory_mb: int = 4096):
        self.timeout_sec = timeout_sec
        self.max_memory_mb = max_memory_mb

    # ------------------------------------------------------------------ #
    # Static worker – must not capture ``self`` so it serialises cleanly
    # under the *spawn* start method.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _worker(
        max_memory_mb: int,
        conn,
        func: Callable,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Target routine executed in the child process."""
        try:
            if sys.platform.startswith("linux"):
                try:
                    import resource

                    max_bytes = max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
                except Exception:
                    pass
            result = func(*args, **kwargs)
            conn.send(("ok", result))
        except MemoryError as exc:
            conn.send(("memory_error", exc))
        except Exception as exc:
            conn.send(("error", exc))
        finally:
            conn.close()

    def run(self, func: Callable, *args, **kwargs) -> Any:
        """Run *func* in a subprocess and return its result.

        Raises
        ------
        TimeoutError
            If the child does not finish within ``self.timeout_sec``.
        MemoryError
            If the child exceeds ``self.max_memory_mb`` or is killed by OOM.
        Exception
            Any exception raised by *func* is re-raised in the parent.
        """
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()

        p = ctx.Process(
            target=self._worker,
            args=(self.max_memory_mb, child_conn, func, args, kwargs),
        )
        p.start()
        child_conn.close()  # parent does not need the child end

        p.join(timeout=self.timeout_sec)

        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()
            parent_conn.close()
            raise TimeoutError(
                f"Sandbox timed out after {self.timeout_sec}s"
            )

        try:
            if parent_conn.poll():
                status, payload = parent_conn.recv()
                parent_conn.close()
                if status == "ok":
                    return payload
                elif status == "memory_error":
                    raise MemoryError(
                        f"Memory limit ({self.max_memory_mb} MB) exceeded"
                    ) from payload
                else:
                    raise payload
        except EOFError:
            parent_conn.close()
            raise MemoryError(
                f"Sandbox process killed, likely OOM (> {self.max_memory_mb} MB)"
            )
