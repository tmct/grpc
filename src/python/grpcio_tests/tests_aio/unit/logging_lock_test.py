# Copyright 2026 The gRPC Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression tests for the AIO/logging lock inversion in #43421."""

import os
import subprocess
import sys
import textwrap
import unittest

# A subprocess bounds a real deadlock and gives each case fresh AIO state.
_REPRODUCER = textwrap.dedent("""\
    import asyncio
    import logging
    import sys
    import threading
    from unittest import mock

    from grpc._cython import cygrpc

    mode = sys.argv[1]
    logger = logging.getLogger("grpc._cython.cygrpc")
    logger.setLevel(logging.WARNING)
    original_debug = logger.debug
    requested = threading.Event()
    lock_held = threading.Event()
    completed = threading.Event()
    main_thread = threading.current_thread()
    errors = []
    intercepted = False

    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if not requested.wait(5):
                raise AssertionError("Initialization did not reach the log")
            # dictConfig holds this lock when GC can enter an AIO finalizer.
            # A balanced init/shutdown pair exercises the same AIO lock without
            # depending on GC timing or transferring ownership of an AIO ref.
            logging._acquireLock()
            try:
                logger._cache.clear()
                lock_held.set()
                cygrpc.init_grpc_aio()
                cygrpc.shutdown_grpc_aio()
            finally:
                logging._releaseLock()
        except BaseException as error:
            errors.append(error)
        finally:
            loop.close()
            completed.set()

    def debug(message, *args, **kwargs):
        global intercepted
        if threading.current_thread() is not main_thread:
            return
        target = {
            "engine": "Using %s as I/O engine",
            "running": "Loaded running loop",
            "policy": "Loaded policy loop",
        }[mode]
        if not intercepted and target in message:
            intercepted = True
            requested.set()
            assert lock_held.wait(5), "Worker did not acquire logging's lock"
        # Even at WARNING, a DEBUG cache miss takes logging's module lock.
        original_debug(message, *args, **kwargs)

    def exercise():
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        with mock.patch.object(logger, "debug", side_effect=debug):
            cygrpc.init_grpc_aio()
            try:
                assert intercepted, "Expected initialization log was not emitted"
                assert completed.wait(5), "AIO initialization blocked the worker"
                assert not errors, errors
            finally:
                cygrpc.shutdown_grpc_aio()
        thread.join(timeout=5)
        assert not thread.is_alive()

    if mode == "running":
        async def main():
            # Keep the process initialized to exercise the per-call path.
            cygrpc.init_grpc_aio()
            try:
                exercise()
            finally:
                cygrpc.shutdown_grpc_aio()
        asyncio.run(main())
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            exercise()
        finally:
            loop.close()
""")


class TestLoggingLock(unittest.TestCase):
    def _run_script(self, script, *args):
        result = subprocess.run(
            [sys.executable, "-c", script, *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_engine_logging(self):
        self._run_script(_REPRODUCER, "engine")

    def test_running_loop_logging(self):
        self._run_script(_REPRODUCER, "running")

    def test_policy_loop_logging(self):
        self._run_script(_REPRODUCER, "policy")

    def test_failed_loop_lookup_preserves_aio_reference(self):
        self._run_script(
            textwrap.dedent("""\
            import asyncio
            from unittest import mock

            from grpc._cython import cygrpc

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            cygrpc.init_grpc_aio()
            try:
                with mock.patch.object(
                    asyncio, "get_running_loop", side_effect=ValueError("lookup")
                ):
                    try:
                        cygrpc.AioChannel(b"localhost:1234", (), None, loop)
                    except ValueError as error:
                        assert str(error) == "lookup"
                    else:
                        raise AssertionError("Loop lookup should have failed")
                # The failed constructor's __dealloc__ must only release its
                # own reference, leaving the explicit reference above alive.
                cygrpc.shutdown_grpc_aio()
            finally:
                loop.close()
        """)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
