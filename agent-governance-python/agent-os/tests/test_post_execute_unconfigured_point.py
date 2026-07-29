# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""``post_execute`` must tell an unconfigured point apart from a denial.

The engine reports an intervention point the manifest does not configure as
``runtime_error:intervention_point_unknown``. That is the right answer to a
request naming an unknown point, but the adapter is not answering a request:
it evaluates output after every call whether or not the host asked for output
governance. Reading the error as a denial would block every response under any
manifest that binds only ``input`` or only ``pre_tool_call``.

The risk in relaxing it is that output enforcement quietly stops working, so
these pin both directions: an unconfigured point permits, and a configured one
still denies, transforms, and records completion the way it did.
"""

from __future__ import annotations

import types

import pytest

from agent_os.integrations.base import BaseIntegration

POINT_UNKNOWN = "runtime_error:intervention_point_unknown"


class _Result:
    """Stand-in for the adapter runtime's result object."""

    def __init__(self, allowed: bool, reason: str | None = None, transform=None):
        self.allowed = allowed
        self.reason = reason
        self.transform = transform


class _Runtime:
    def __init__(self, result: _Result) -> None:
        self._result = result
        self.calls = 0

    def evaluate_output(self, state, *, content):
        self.calls += 1
        return self._result


def _adapter(result: _Result):
    """Build a bare adapter bound to a scripted runtime."""
    adapter = object.__new__(BaseIntegration)
    adapter._adapter_runtime = _Runtime(result)
    adapter.completed = []
    adapter.record_host_completion = lambda state, **kw: adapter.completed.append(kw)
    return adapter


STATE = types.SimpleNamespace(agent_id="a", session_id="s")


def test_unconfigured_output_point_permits():
    adapter = _adapter(_Result(allowed=False, reason=POINT_UNKNOWN))

    allowed, reason = adapter.post_execute(STATE, "the model's answer")

    assert (allowed, reason) == (True, None)
    # Completion still has to be recorded, or budgets drift on every manifest
    # that does not bind output.
    assert adapter.completed == [{"output_data": "the model's answer"}]


def test_configured_output_point_still_denies():
    """The relaxation must not extend to a real denial."""
    adapter = _adapter(_Result(allowed=False, reason="blocked_pattern_output"))

    allowed, reason = adapter.post_execute(STATE, "leaked secret")

    assert allowed is False
    assert reason == "blocked_pattern_output"
    assert adapter.completed == []


def test_configured_output_transform_is_still_refused():
    """A transform carries a replacement this contract cannot return."""
    adapter = _adapter(
        _Result(allowed=True, reason=None, transform=object())
    )

    allowed, reason = adapter.post_execute(STATE, "card 4111111111111111")

    assert allowed is False
    assert reason == "transform_not_applicable"


def test_configured_output_allow_passes():
    adapter = _adapter(_Result(allowed=True))

    allowed, reason = adapter.post_execute(STATE, "fine")

    assert (allowed, reason) == (True, None)
    assert adapter.completed == [{"output_data": "fine"}]


@pytest.mark.parametrize(
    "reason",
    ["runtime_error:manifest_invalid", "runtime_error:path_missing", "denied"],
)
def test_other_runtime_errors_still_deny(reason):
    """Only the unconfigured-point reason is relaxed."""
    adapter = _adapter(_Result(allowed=False, reason=reason))

    allowed, got = adapter.post_execute(STATE, "x")

    assert allowed is False
    assert got == reason
