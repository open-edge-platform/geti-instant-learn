# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Guard: the data/model contract must import without torch installed."""

import subprocess  # noqa: S404
import sys
import textwrap

_GUARD_SCRIPT = textwrap.dedent(
    """
    import builtins
    _real_import = builtins.__import__
    def _blocked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked (attempted: " + name + ")")
        return _real_import(name, *args, **kwargs)
    # Block torch before importing anything from instantlearn so that the real
    # import machinery — including every package __init__ on the path — is
    # exercised under the no-torch constraint.
    builtins.__import__ = _blocked_import

    import numpy as np

    # Public import paths for the backend-neutral contract. These traverse
    # instantlearn.data.__init__ and instantlearn.data.base.__init__, so any
    # torch import leaked by those package __init__ files would fail here.
    from instantlearn.data.base import Batch, Category, Prediction, Sample

    pred = Prediction(
        masks=np.zeros((1, 4, 4), dtype=np.uint8),
        scores=np.ones((1,), dtype=np.float32),
        label_ids=np.zeros((1,), dtype=np.int32),
        label_names=np.array(["object"], dtype=object),
    )
    assert pred.masks.shape == (1, 4, 4)
    assert pred.scores.dtype.name == "float32"

    sample = Sample(categories=[Category(id=0, label="object")])
    batch = Batch.collate([sample])
    assert batch[0].category_labels == ["object"]
    print("OK")
""",
)


def test_contract_imports_without_torch() -> None:
    """The data contract must import and construct without torch.

    Runs the verification in a subprocess with torch blocked via an
    ``__import__`` hook. The contract types (``Sample``, ``Batch``,
    ``Category``, ``Prediction``) are imported through their public
    ``instantlearn.data.base`` path so the real import machinery — including
    the ``instantlearn.data`` and ``instantlearn.data.base`` package
    ``__init__`` files — is exercised under the no-torch constraint.
    Torch-bound members such as ``Dataset`` are loaded lazily and are not
    touched here.
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
