# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Guard: the data/model contract must import without torch installed."""

import subprocess  # noqa: S404
import sys
import textwrap

_GUARD_SCRIPT = textwrap.dedent(
    """
    import builtins
    import importlib.util
    import pathlib
    import sys
    _real_import = builtins.__import__
    def _blocked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked (attempted: " + name + ")")
        return _real_import(name, *args, **kwargs)
    # Locate the instantlearn source root BEFORE blocking torch
    _spec = importlib.util.find_spec("instantlearn")
    _src_root = pathlib.Path(_spec.submodule_search_locations[0]).parent
    # Block torch from this point on
    builtins.__import__ = _blocked_import
    def _load_module(mod_name, rel_path):
        path = _src_root / rel_path
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod
    _load_module("instantlearn.data.base.prediction", "instantlearn/data/base/prediction.py")
    from instantlearn.data.base.prediction import Prediction
    import numpy as np
    pred = Prediction(
        masks=np.zeros((1, 4, 4), dtype=np.uint8),
        scores=np.ones((1,), dtype=np.float32),
        label_ids=np.zeros((1,), dtype=np.int32),
        label_names=np.array(["object"], dtype=object),
    )
    assert pred.masks.shape == (1, 4, 4)
    assert pred.scores.dtype.name == "float32"
    print("OK")
""",
)


def test_contract_imports_without_torch() -> None:
    """The data/model contract must import and construct without torch.

    Runs the verification in a subprocess with torch blocked via an
    ``__import__`` hook. ``instantlearn.models.torch_base`` and
    ``instantlearn.models.openvino_base`` are intentionally excluded
    (they legitimately need torch / openvino respectively).
    """
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _GUARD_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "Contract import failed with torch blocked.\nstdout: " + result.stdout + "\nstderr: " + result.stderr
    )
    assert "OK" in result.stdout
