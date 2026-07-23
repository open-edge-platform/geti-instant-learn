# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`~instantlearn.models.torch_adapter.CategoryRegistry`.

Covers every public factory method and protocol implementation, plus regression
tests for the two attribute-name bugs that were fixed:

* ``Matcher.to_openvino()`` referenced ``self._category_names`` (AttributeError).
* ``GroundedSAM.predict()`` referenced ``self.category_mapping`` (AttributeError).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from instantlearn.data.base.batch import Batch
from instantlearn.data.base.sample import Category, Sample
from instantlearn.models.torch_adapter import CategoryRegistry, TensorSample


def _make_sample(label_ids: list[int], labels: list[str]) -> Sample:
    return Sample(categories=[Category(id=i, label=l) for i, l in zip(label_ids, labels, strict=True)])


def _make_tensor_sample(label_ids: list[int], labels: list[str]) -> TensorSample:
    return TensorSample(
        category_labels=labels,
        label_ids=torch.tensor(label_ids, dtype=torch.int32) if label_ids else None,
    )


class TestFromSamples:
    def test_single_sample(self) -> None:
        sample = _make_sample([0, 1], ["cat", "dog"])
        reg = CategoryRegistry.from_samples(sample)
        assert reg.id_to_name == {0: "cat", 1: "dog"}
        assert reg.name_to_id == {"cat": 0, "dog": 1}

    def test_batch(self) -> None:
        s1 = _make_sample([0], ["cat"])
        s2 = _make_sample([1], ["dog"])
        batch = Batch.collate([s1, s2])
        reg = CategoryRegistry.from_samples(batch)
        assert reg.id_to_name == {0: "cat", 1: "dog"}

    def test_list_of_tensor_samples(self) -> None:
        ts = [
            _make_tensor_sample([2], ["shoe"]),
            _make_tensor_sample([3], ["hat"]),
        ]
        reg = CategoryRegistry.from_samples(ts)
        assert reg.id_to_name == {2: "shoe", 3: "hat"}

    def test_first_occurrence_wins_on_duplicate_name(self) -> None:
        """Same name with conflicting ids — the first id assignment wins."""
        s1 = _make_sample([0], ["cat"])
        s2 = _make_sample([99], ["cat"])  # same name, different id
        reg = CategoryRegistry.from_samples([s1, s2])
        assert reg.name_to_id["cat"] == 0  # first wins

    def test_first_occurrence_wins_on_duplicate_id(self) -> None:
        """Same id with different names — from_samples keeps first by setdefault logic."""
        s1 = _make_sample([0], ["cat"])
        s2 = _make_sample([0], ["kitten"])
        reg = CategoryRegistry.from_samples([s1, s2])
        assert reg.id_to_name[0] == "cat"  # first wins

    def test_empty_sample_list(self) -> None:
        reg = CategoryRegistry.from_samples([])
        assert len(reg) == 0

    def test_sample_without_categories(self) -> None:
        """A sample with no explicit categories carries a default 'object' category."""
        sample = Sample(image=np.zeros((10, 10, 3), dtype=np.uint8))
        reg = CategoryRegistry.from_samples(sample)
        assert reg.id_to_name == {0: "object"}


class TestFromMetadata:
    def test_int_keys(self) -> None:
        reg = CategoryRegistry.from_metadata({0: "cat", 1: "dog"})
        assert reg.id_to_name == {0: "cat", 1: "dog"}
        assert reg.name_to_id == {"cat": 0, "dog": 1}

    def test_string_keys(self) -> None:
        """JSON object keys are strings; from_metadata must convert them."""
        reg = CategoryRegistry.from_metadata({"0": "cat", "1": "dog"})
        assert reg.id_to_name == {0: "cat", 1: "dog"}

    def test_empty_mapping(self) -> None:
        reg = CategoryRegistry.from_metadata({})
        assert len(reg) == 0


class TestFromNames:
    def test_default_start_id(self) -> None:
        reg = CategoryRegistry.from_names(["cat", "dog", "bird"])
        assert reg.id_to_name == {0: "cat", 1: "dog", 2: "bird"}
        assert reg.name_to_id == {"cat": 0, "dog": 1, "bird": 2}

    def test_custom_start_id(self) -> None:
        reg = CategoryRegistry.from_names(["a", "b"], start_id=1)
        assert reg.id_to_name == {1: "a", 2: "b"}
        assert reg.name_to_id == {"a": 1, "b": 2}

    def test_empty_list(self) -> None:
        reg = CategoryRegistry.from_names([])
        assert len(reg) == 0

    def test_single_name(self) -> None:
        reg = CategoryRegistry.from_names(["only"])
        assert reg[0] == "only"


class TestMerge:
    def test_disjoint_registries(self) -> None:
        base = CategoryRegistry.from_names(["cat"], start_id=0)
        other = CategoryRegistry.from_names(["dog"], start_id=1)
        merged = base.merge(other)
        assert merged.id_to_name == {0: "cat", 1: "dog"}

    def test_other_wins_on_overlap(self) -> None:
        base = CategoryRegistry.from_metadata({0: "old_name"})
        other = CategoryRegistry.from_metadata({0: "new_name"})
        merged = base.merge(other)
        assert merged.id_to_name[0] == "new_name"

    def test_base_entries_preserved_for_non_overlapping_ids(self) -> None:
        base = CategoryRegistry.from_metadata({0: "cat", 1: "dog"})
        other = CategoryRegistry.from_metadata({1: "puppy"})
        merged = base.merge(other)
        assert merged.id_to_name[0] == "cat"
        assert merged.id_to_name[1] == "puppy"

    def test_merge_with_empty_other_is_identity(self) -> None:
        base = CategoryRegistry.from_names(["cat", "dog"])
        merged = base.merge(CategoryRegistry())
        assert merged.id_to_name == base.id_to_name
        assert merged.name_to_id == base.name_to_id

    def test_merge_into_empty_base(self) -> None:
        other = CategoryRegistry.from_names(["cat"])
        merged = CategoryRegistry().merge(other)
        assert merged.id_to_name == {0: "cat"}

    def test_original_registries_not_mutated(self) -> None:
        base = CategoryRegistry.from_names(["cat"])
        other = CategoryRegistry.from_names(["dog"], start_id=1)
        _ = base.merge(other)
        assert len(base) == 1
        assert len(other) == 1


class TestNamesIndexed:
    def test_contiguous_ids(self) -> None:
        reg = CategoryRegistry.from_names(["cat", "dog", "bird"])
        assert reg.names_indexed() == ["cat", "dog", "bird"]

    def test_gap_falls_back_to_str(self) -> None:
        reg = CategoryRegistry.from_metadata({0: "cat", 2: "dog"})
        assert reg.names_indexed() == ["cat", "1", "dog"]

    def test_empty_registry(self) -> None:
        assert CategoryRegistry().names_indexed() == []


class TestMappingProtocol:
    def test_getitem(self) -> None:
        reg = CategoryRegistry.from_names(["cat"])
        assert reg[0] == "cat"

    def test_getitem_missing_raises(self) -> None:
        reg = CategoryRegistry()
        with pytest.raises(KeyError):
            _ = reg[99]

    def test_iter_yields_ids(self) -> None:
        reg = CategoryRegistry.from_names(["a", "b", "c"])
        assert sorted(reg) == [0, 1, 2]

    def test_len(self) -> None:
        reg = CategoryRegistry.from_names(["x", "y"])
        assert len(reg) == 2

    def test_get_with_default(self) -> None:
        reg = CategoryRegistry.from_names(["cat"])
        assert reg.get(0) == "cat"
        assert reg.get(99, "unknown") == "unknown"

    def test_bool_empty(self) -> None:
        assert not CategoryRegistry()

    def test_bool_non_empty(self) -> None:
        assert CategoryRegistry.from_names(["cat"])


class TestRepr:
    def test_repr_contains_class_name(self) -> None:
        reg = CategoryRegistry.from_names(["cat", "dog"])
        r = repr(reg)
        assert r.startswith("CategoryRegistry(")

    def test_repr_contains_pairs(self) -> None:
        reg = CategoryRegistry.from_metadata({1: "cat"})
        assert "1: 'cat'" in repr(reg)

    def test_repr_empty(self) -> None:
        assert repr(CategoryRegistry()) == "CategoryRegistry({})"
