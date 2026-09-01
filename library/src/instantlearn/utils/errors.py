# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions for instantlearn."""


class ModelNotFittedError(RuntimeError):
    """Raised when ``predict()`` is called before ``fit()`` on models that require reference data.

    Models that support visual-exemplar or few-shot prompting (e.g. Matcher, PerDino,
    SAM3 in visual-exemplar mode) store reference state set by ``fit()``. Calling
    ``predict()`` on such a model before ``fit()`` raises this error.

    Example:
        >>> model = Matcher()
        >>> model.predict(target)  # raises ModelNotFittedError
        >>> model.fit(reference)
        >>> model.predict(target)  # OK
    """
