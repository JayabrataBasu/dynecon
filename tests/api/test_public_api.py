"""The public API is a literal set: changing it requires editing this test deliberately."""

import econkit

EXPECTED = {
    "AR1Shock",
    "Control",
    "Model",
    "Param",
    "Solution",
    "State",
    "__version__",
    "backends",
    "config",
    "errors",
    "simulate",
    "solve",
}


def test_public_api_surface() -> None:
    assert set(econkit.__all__) == EXPECTED
    for name in EXPECTED:
        assert hasattr(econkit, name)
