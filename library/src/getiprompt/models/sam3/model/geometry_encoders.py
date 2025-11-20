# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from typing_extensions import override

from .act_ckpt_utils import activation_ckpt_wrapper
from .box_ops import box_cxcywh_to_xyxy

from .model_misc import get_clones


def is_right_padded(mask):
    """Given a padding mask (following pytorch convention, 1s for padded values),
    returns whether the padding is on the right or not."""
    return (mask.long() == torch.sort(mask.long(), dim=-1)[0]).all()


def concat_padded_sequences(seq1, mask1, seq2, mask2, return_index: bool = False):
    """
    Concatenates two right-padded sequences, such that the resulting sequence
    is contiguous and also right-padded.

    Following pytorch's convention, tensors are sequence first, and the mask are
    batch first, with 1s for padded values.

    :param seq1: A tensor of shape (seq1_length, batch_size, hidden_size).
    :param mask1: A tensor of shape (batch_size, seq1_length).
    :param seq2: A tensor of shape (seq2_length, batch_size,  hidden_size).
    :param mask2: A tensor of shape (batch_size, seq2_length).
    :param return_index: If True, also returns the index of the ids of the element of seq2
        in the concatenated sequence. This can be used to retrieve the elements of seq2
    :return: A tuple (concatenated_sequence, concatenated_mask) if return_index is False,
        otherwise (concatenated_sequence, concatenated_mask, index).
    """
    seq1_length, batch_size, hidden_size = seq1.shape
    seq2_length, batch_size, hidden_size = seq2.shape

    assert batch_size == seq1.size(1) == seq2.size(1) == mask1.size(0) == mask2.size(0)
    assert hidden_size == seq1.size(2) == seq2.size(2)
    assert seq1_length == mask1.size(1)
    assert seq2_length == mask2.size(1)

    torch._assert_async(is_right_padded(mask1))
    torch._assert_async(is_right_padded(mask2))

    actual_seq1_lengths = (~mask1).sum(dim=-1)
    actual_seq2_lengths = (~mask2).sum(dim=-1)

    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length
    concatenated_mask = (
        torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1)
        >= final_lengths[:, None]
    )

    # (max_len, batch_size, hidden_size)
    concatenated_sequence = torch.zeros(
        (max_length, batch_size, hidden_size), device=seq2.device, dtype=seq2.dtype
    )
    concatenated_sequence[:seq1_length, :, :] = seq1

    # At this point, the element of seq1 are in the right place
    # We just need to shift the elements of seq2

    index = torch.arange(seq2_length, device=seq2.device)[:, None].repeat(1, batch_size)
    index = index + actual_seq1_lengths[None]

    concatenated_sequence = concatenated_sequence.scatter(
        0, index[:, :, None].expand(-1, -1, hidden_size), seq2
    )

    if return_index:
        return concatenated_sequence, concatenated_mask, index

    return concatenated_sequence, concatenated_mask


class Prompt:
    """Utility class to manipulate geometric prompts.

    We expect the sequences in pytorch convention, that is sequence first, batch second
    The dimensions are expected as follows:
    box_embeddings shape: N_boxes x B x C_box
    box_mask shape: B x N_boxes. Can be None if nothing is masked out
    point_embeddings shape: N_points x B x C_point
    point_mask shape: B x N_points. Can be None if nothing is masked out
    mask_embeddings shape: N_masks x B x 1 x H_mask x W_mask
    mask_mask shape: B x N_masks. Can be None if nothing is masked out

    We also store positive/negative labels. These tensors are also stored batch-first
    If they are None, we'll assume positive labels everywhere
    box_labels: long tensor of shape N_boxes x B
    point_labels: long tensor of shape N_points x B
    mask_labels: long tensor of shape N_masks x B
    """

    def __init__(
        self,
        box_embeddings=None,
        box_mask=None,
        point_embeddings=None,
        point_mask=None,
        box_labels=None,
        point_labels=None,
        mask_embeddings=None,
        mask_mask=None,  # Attention mask for mask prompt
        mask_labels=None,
    ):
        # Check for null prompt
        if (
            box_embeddings is None
            and point_embeddings is None
            and mask_embeddings is None
        ):
            self.box_embeddings = None
            self.box_labels = None
            self.box_mask = None
            self.point_embeddings = None
            self.point_labels = None
            self.point_mask = None
            self.mask_embeddings = None
            self.mask_mask = None
            # Masks are assumed positive only for now.
            self.mask_labels = None
            return
        # Get sequence lengths and device
        box_seq_len, point_seq_len, mask_seq_len, bs, device = (
            self._init_seq_len_and_device(
                box_embeddings, point_embeddings, mask_embeddings
            )
        )

        # Initialize embeds, labels, attention masks.
        box_embeddings, box_labels, box_mask = self._init_box(
            box_embeddings, box_labels, box_mask, box_seq_len, bs, device
        )
        point_embeddings, point_labels, point_mask = self._init_point(
            point_embeddings, point_labels, point_mask, point_seq_len, bs, device
        )
        mask_embeddings, mask_labels, mask_mask = self._init_mask(
            mask_embeddings, mask_labels, mask_mask, mask_seq_len, bs, device
        )

        # Dimension checks
        assert (
            box_embeddings is not None
            and list(box_embeddings.shape[:2])
            == [
                box_seq_len,
                bs,
            ]
        ), f"Wrong dimension for box embeddings. Expected [{box_seq_len}, {bs}, *] got {box_embeddings.shape}"
        assert (
            box_mask is not None
            and list(box_mask.shape)
            == [
                bs,
                box_seq_len,
            ]
        ), f"Wrong dimension for box mask. Expected [{bs}, {box_seq_len}] got {box_mask.shape}"
        assert (
            point_embeddings is not None
            and list(point_embeddings.shape[:2])
            == [
                point_seq_len,
                bs,
            ]
        ), f"Wrong dimension for point embeddings. Expected [{point_seq_len}, {bs}, *] got {point_embeddings.shape}"
        assert (
            point_mask is not None
            and list(point_mask.shape)
            == [
                bs,
                point_seq_len,
            ]
        ), f"Wrong dimension for point mask. Expected [{bs}, {point_seq_len}] got {point_mask.shape}"
        assert (
            box_labels is not None
            and list(box_labels.shape)
            == [
                box_seq_len,
                bs,
            ]
        ), f"Wrong dimension for box labels. Expected [{box_seq_len}, {bs}] got {box_labels.shape}"
        assert (
            point_labels is not None
            and list(point_labels.shape)
            == [
                point_seq_len,
                bs,
            ]
        ), f"Wrong dimension for point labels. Expected [{point_seq_len}, {bs}] got {point_labels.shape}"
        assert (
            # Allowed to be None, we leave it to the encoder to check for validity before encoding.
            mask_embeddings is None
            or list(mask_embeddings.shape[:2])
            == [
                mask_seq_len,
                bs,
            ]
        ), f"Wrong dimension for mask embeddings. Expected [{mask_seq_len}, {bs}, *] got {mask_embeddings.shape}"
        assert (
            mask_mask is None
            or list(mask_mask.shape)
            == [
                bs,
                mask_seq_len,
            ]
        ), f"Wrong dimension for mask attn. mask. Expected [{bs}, {mask_seq_len}] got {mask_mask.shape}"

        # Device checks
        assert (
            box_embeddings is not None and box_embeddings.device == device
        ), f"Expected box embeddings to be on device {device}, got {box_embeddings.device}"
        assert (
            box_mask is not None and box_mask.device == device
        ), f"Expected box mask to be on device {device}, got {box_mask.device}"
        assert (
            box_labels is not None and box_labels.device == device
        ), f"Expected box labels to be on device {device}, got {box_labels.device}"
        assert (
            point_embeddings is not None and point_embeddings.device == device
        ), f"Expected point embeddings to be on device {device}, got {point_embeddings.device}"
        assert (
            point_mask is not None and point_mask.device == device
        ), f"Expected point mask to be on device {device}, got {point_mask.device}"
        assert (
            point_labels is not None and point_labels.device == device
        ), f"Expected point labels to be on device {device}, got {point_labels.device}"
        assert (
            mask_embeddings is None or mask_embeddings.device == device
        ), f"Expected mask embeddings to be on device {device}, got {mask_embeddings.device}"
        assert (
            mask_mask is None or mask_mask.device == device
        ), f"Expected mask attn. mask to be on device {device}, got {mask_mask.device}"

        self.box_embeddings = box_embeddings
        self.point_embeddings = point_embeddings
        self.box_mask = box_mask
        self.point_mask = point_mask
        self.box_labels = box_labels
        self.point_labels = point_labels
        self.mask_embeddings = mask_embeddings
        self.mask_labels = mask_labels
        self.mask_mask = mask_mask

    def _init_seq_len_and_device(
        self, box_embeddings, point_embeddings, mask_embeddings
    ):
        box_seq_len = point_seq_len = mask_seq_len = 0
        bs = None
        device = None
        if box_embeddings is not None:
            bs = box_embeddings.shape[1]
            box_seq_len = box_embeddings.shape[0]
            device = box_embeddings.device

        if point_embeddings is not None:
            point_seq_len = point_embeddings.shape[0]
            if bs is not None:
                assert (
                    bs == point_embeddings.shape[1]
                ), f"Batch size mismatch between box and point embeddings. Got {bs} and {point_embeddings.shape[1]}."
            else:
                bs = point_embeddings.shape[1]
            if device is not None:
                assert (
                    device == point_embeddings.device
                ), "Device mismatch between box and point embeddings"
            else:
                device = point_embeddings.device

        if mask_embeddings is not None:
            mask_seq_len = mask_embeddings.shape[0]
            if bs is not None:
                assert (
                    bs == mask_embeddings.shape[1]
                ), f"Batch size mismatch between box/point and mask embedding. Got {bs} and {mask_embeddings.shape[1]}"
            else:
                bs = mask_embeddings.shape[1]
            if device is not None:
                assert (
                    device == mask_embeddings.device
                ), "Device mismatch between box/point and mask embeddings."
            else:
                device = mask_embeddings.device

        return box_seq_len, point_seq_len, mask_seq_len, bs, device

    def _init_box(self, box_embeddings, box_labels, box_mask, box_seq_len, bs, device):
        if box_embeddings is None:
            box_embeddings = torch.zeros(box_seq_len, bs, 4, device=device)
        if box_labels is None:
            box_labels = torch.ones(box_seq_len, bs, device=device, dtype=torch.long)
        if box_mask is None:
            box_mask = torch.zeros(bs, box_seq_len, device=device, dtype=torch.bool)
        return box_embeddings, box_labels, box_mask

    def _init_point(
        self, point_embeddings, point_labels, point_mask, point_seq_len, bs, device
    ):
        """
        Identical to _init_box. Except that C=2 for points (vs. 4 for boxes).
        """
        if point_embeddings is None:
            point_embeddings = torch.zeros(point_seq_len, bs, 2, device=device)
        if point_labels is None:
            point_labels = torch.ones(
                point_seq_len, bs, device=device, dtype=torch.long
            )
        if point_mask is None:
            point_mask = torch.zeros(bs, point_seq_len, device=device, dtype=torch.bool)
        return point_embeddings, point_labels, point_mask

    def _init_mask(
        self, mask_embeddings, mask_labels, mask_mask, mask_seq_len, bs, device
    ):
        # NOTE: Mask embeddings can be of arbitrary resolution, so we don't initialize it here.
        # In case we append new mask, we check that its resolution matches exisiting ones (if any).
        # In case mask_embeddings is None, we should never encode it.
        if mask_labels is None:
            mask_labels = torch.ones(mask_seq_len, bs, device=device, dtype=torch.long)
        if mask_mask is None:
            mask_mask = torch.zeros(bs, mask_seq_len, device=device, dtype=torch.bool)
        return mask_embeddings, mask_labels, mask_mask

    def append_boxes(self, boxes, labels, mask=None):
        if self.box_embeddings is None:
            self.box_embeddings = boxes
            self.box_labels = labels
            self.box_mask = mask
            return

        bs = self.box_embeddings.shape[1]
        assert boxes.shape[1] == labels.shape[1] == bs
        assert list(boxes.shape[:2]) == list(labels.shape[:2])
        if mask is None:
            mask = torch.zeros(
                bs, boxes.shape[0], dtype=torch.bool, device=boxes.device
            )

        self.box_labels, _ = concat_padded_sequences(
            self.box_labels.unsqueeze(-1), self.box_mask, labels.unsqueeze(-1), mask
        )
        self.box_labels = self.box_labels.squeeze(-1)
        self.box_embeddings, self.box_mask = concat_padded_sequences(
            self.box_embeddings, self.box_mask, boxes, mask
        )

    def append_points(self, points, labels, mask=None):
        if self.point_embeddings is None:
            self.point_embeddings = points
            self.point_labels = labels
            self.point_mask = mask
            return

        bs = self.point_embeddings.shape[1]
        assert points.shape[1] == labels.shape[1] == bs
        assert list(points.shape[:2]) == list(labels.shape[:2])
        if mask is None:
            mask = torch.zeros(
                bs, points.shape[0], dtype=torch.bool, device=points.device
            )

        self.point_labels, _ = concat_padded_sequences(
            self.point_labels.unsqueeze(-1), self.point_mask, labels.unsqueeze(-1), mask
        )
        self.point_labels = self.point_labels.squeeze(-1)
        self.point_embeddings, self.point_mask = concat_padded_sequences(
            self.point_embeddings, self.point_mask, points, mask
        )

    def append_masks(self, masks, labels=None, attn_mask=None):
        if labels is not None:
            assert list(masks.shape[:2]) == list(labels.shape[:2])
        if self.mask_embeddings is None:
            self.mask_embeddings = masks
            mask_seq_len, bs = masks.shape[:2]
            if labels is None:
                self.mask_labels = torch.ones(
                    mask_seq_len, bs, device=masks.device, dtype=torch.long
                )
            else:
                self.mask_labels = labels
            if attn_mask is None:
                self.mask_mask = torch.zeros(
                    bs, mask_seq_len, device=masks.device, dtype=torch.bool
                )
            else:
                self.mask_mask = attn_mask
        else:
            raise NotImplementedError("Only one mask per prompt is supported.")

    def clone(self):
        return Prompt(
            box_embeddings=(
                None if self.box_embeddings is None else self.box_embeddings.clone()
            ),
            box_mask=None if self.box_mask is None else self.box_mask.clone(),
            point_embeddings=(
                None if self.point_embeddings is None else self.point_embeddings.clone()
            ),
            point_mask=None if self.point_mask is None else self.point_mask.clone(),
            box_labels=None if self.box_labels is None else self.box_labels.clone(),
            point_labels=(
                None if self.point_labels is None else self.point_labels.clone()
            ),
        )
