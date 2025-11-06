# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Geti Prompt."""

import colorsys
import hashlib
import logging
import random
import sys
from itertools import starmap
from pathlib import Path

import cv2
import numpy as np
import PIL
import requests
import torch
import umap
from matplotlib import pyplot as plt
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.nn import functional
from torchvision import transforms

from getiprompt.types import Masks

logger = logging.getLogger("Geti Prompt")


def setup_logger(dir_path: Path | None = None, log_level: str = "INFO") -> None:
    """Save logs to a directory and setup console logging."""
    logger = logging.getLogger("Geti Prompt")
    logger.setLevel(log_level.upper())
    logger.propagate = False  # This will prevent duplicate logs

    # Clear existing handlers to prevent duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if dir_path:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(dir_path / "logs.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s: \t%(message)s"))
    logger.addHandler(console_handler)

    # Set other loggers to a higher level to avoid verbose debug logs
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("sam2").setLevel(logging.WARNING)


def precision_to_torch_dtype(precision: str) -> torch.dtype:
    """Convert a precision string to a torch.dtype."""
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision.lower()]


def download_file(url: str, target_path: Path, sha_sum: str | None = None) -> None:
    """Download a file from a URL to a target path.

    Args:
        url: URL to download the file from
        target_path: Path to save the file to
        sha_sum: SHA-256 checksum of the file
    """
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    disable_progress = not sys.stderr.isatty()
    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        " • ",
        DownloadColumn(),
        " • ",
        TransferSpeedColumn(),
        " • ",
        TimeRemainingColumn(),
        transient=True,
        disable=disable_progress,
    )

    try:  # noqa: PLR1702
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            msg = f"Downloading {target_path.name} ({total_size / (1024 * 1024):.2f} MB) from {url}..."
            logger.info(msg)

            with progress:
                task_id = progress.add_task("download", total=total_size, filename=target_path.name)
                with target_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(task_id, advance=len(chunk))

            if not disable_progress and total_size > 0:
                progress.update(task_id, completed=total_size)

        if sha_sum:
            check_file_hash(target_path, sha_sum)

        msg = f"Downloaded model weights successfully to {target_path}"
        logger.info(msg)
    except Exception:
        logger.exception("An unexpected error occurred during download.")
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                msg = f"Error removing file {target_path} after error"
                logger.exception(msg)
        raise


def check_file_hash(file_path: Path, expected_hash: str) -> None:
    """Check if the file hash matches the expected hash.

    Args:
        file_path: Path to the file to check the hash of
        expected_hash: Expected SHA-256 hash of the file

    Raises:
        ValueError: If the file hash does not match the expected hash
    """
    file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if file_hash != expected_hash:
        msg = f"File {file_path} has incorrect hash. Expected {expected_hash}, got {file_hash}"
        raise ValueError(msg)


def compute_file_hash(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file.

    Args:
        file_path: Path to the file to compute the hash of

    Raises:
        ValueError: If the file does not exist

    Returns:
        SHA-256 hash of the file
    """
    if not file_path.exists():
        msg = f"Filepath {file_path} does not exist."
        raise ValueError(msg)
    hash_obj = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):  # Read in 64 KB chunks
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def get_colors(n: int) -> np.ndarray:
    """Generate colors for a mask.

    Args:
        n: Number of colors to generate

    Returns:
        Colors for a mask
    """
    hsv_tuples = [(x / n, 0.5, 0.5) for x in range(n)]
    rgb_tuples = starmap(colorsys.hsv_to_rgb, hsv_tuples)
    return (np.array(list(rgb_tuples)) * 255).astype(np.uint8)


def prepare_target_guided_prompting(
    sim: torch.Tensor,
    reference_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare target guided prompting for the decoder.

    Produces attention similarity and target embedding for the decoder.
    This technique is used in Per-Segment-Anything.
    It can improve the performance of the decoder by providing additional information.
    Note that not all backbones support this technique.

    Args:
        sim: similarity tensor
        reference_features: reference features tensor

    Returns:
        attention_similarity: attention similarity tensor
        reference_features: reference features tensor
    """
    # For multiple similarity masks (e.g. Part-level features), we take the mean of the similarity maps
    if len(sim.shape) == 3:
        sim = sim.mean(dim=0)

    sim = (sim - sim.mean()) / torch.std(sim)
    sim = functional.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    attention_similarity = sim.sigmoid_().unsqueeze(0).flatten(3)

    # For multiple reference features (e.g. Part-level features), we take the mean of the reference features
    if len(reference_features.shape) == 2:
        reference_features = reference_features.mean(0).unsqueeze(0)
    return attention_similarity, reference_features


def cluster_features(
    reference_features: torch.Tensor,
    n_clusters: int = 8,
    visualize: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, plt.Figure]:
    """Create part-level features from reference features.

    This performs a k-means++ clustering on the reference features and takes the centroid as prototype.
    Resulting part-level features are normalized to unit length.
    If n_clusters is 1, the mean of the reference features is taken as prototype.

    Args:
        reference_features: Reference features tensor [X, 256]
        n_clusters: Number of clusters to create (e.g. number of part-level-features)
        visualize: Whether to return UMAP visualization of features and clusters

    Returns:
        Part-level features tensor [n_clusters, 256] and optionally a matplotlib figure
    """
    if n_clusters == 1:
        part_level_features = reference_features.mean(0).unsqueeze(0)
        part_level_features /= part_level_features.norm(
            dim=-1,
            keepdim=True,
        )
        return part_level_features.unsqueeze(0), None  # 1, 256

    features_np = reference_features.cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
    cluster = kmeans.fit_predict(features_np)
    part_level_features = []

    for c in range(n_clusters):
        # use centroid of cluster as prototype
        part_level_feature = features_np[cluster == c].mean(axis=0)
        part_level_feature /= np.linalg.norm(
            part_level_feature,
            axis=-1,
            keepdims=True,
        )
        part_level_features.append(torch.from_numpy(part_level_feature))

    part_level_features = torch.stack(
        part_level_features,
        dim=0,
    ).cuda()  # [n_clusters, 256]

    if visualize:
        features_np = reference_features.cpu().numpy()
        fig = visualize_feature_clusters(
            features=features_np,
            cluster_labels=cluster,
            cluster_centers=kmeans.cluster_centers_,
        )
        return part_level_features, fig

    return part_level_features, None


def cluster_points(points: np.ndarray, n_clusters: int = 8) -> np.ndarray:
    """Cluster points using k-means++."""
    if len(points) < n_clusters:
        return points
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
    cluster = kmeans.fit_predict(points)
    # use centroid of cluster as prototype
    prototypes = []
    for c in range(n_clusters):
        prototype = points[cluster == c].mean(axis=0)
        prototypes.append(prototype)
    return np.array(prototypes).astype(np.int64)


def gen_colors(n: int) -> np.ndarray:
    """Generate colors for a mask.

    Args:
        n: Number of colors to generate

    Returns:
        Colors for a mask
    """
    hsv_tuples = [(x / n, 0.5, 0.5) for x in range(n)]
    rgb_tuples = starmap(colorsys.hsv_to_rgb, hsv_tuples)
    colors = [(0, 0, 0), *list(rgb_tuples)]
    return (np.array(colors) * 255).astype(np.uint8)


def color_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Color overlay of a mask on an image.

    Args:
        image: Image to overlay mask on
        mask: Mask to overlay

    Returns:
        Color overlayed image
    """
    mask_colors = gen_colors(np.max(mask))
    color_mask = mask_colors[mask]  # create color map
    color_mask[mask == 0] = image[mask == 0]  # set background to original color
    image_vis = cv2.addWeighted(image, 0.2, color_mask, 0.8, 0)
    return image_vis[:, :, ::-1]  # BGR2RGB


def visualize_feature_clusters(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
) -> plt.Figure:
    """Create UMAP and t-SNE visualizations of feature clusters side by side.

    Args:
        features: Feature vectors to visualize [N, D]
        cluster_labels: Cluster assignment for each feature [N]
        cluster_centers: Cluster centroids [K, D]

    Returns:
        matplotlib Figure with UMAP and t-SNE visualizations
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # UMAP embedding
    reducer_umap = umap.UMAP(random_state=42)
    embedding_umap = reducer_umap.fit_transform(features)
    centroids_embedding_umap = reducer_umap.transform(cluster_centers)

    # t-SNE embedding with adjusted perplexity
    n_centers = len(cluster_centers)
    perplexity = min(30, n_centers - 1)  # Ensure perplexity is less than n_samples
    reducer_tsne = TSNE(random_state=42, perplexity=perplexity)
    embedding_tsne = reducer_tsne.fit_transform(features)
    centroids_embedding_tsne = reducer_tsne.fit_transform(cluster_centers)

    # Plot UMAP
    scatter1 = ax1.scatter(
        embedding_umap[:, 0],
        embedding_umap[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.6,
        s=100,
    )
    ax1.scatter(
        centroids_embedding_umap[:, 0],
        centroids_embedding_umap[:, 1],
        marker="*",
        s=300,
        c=range(len(cluster_centers)),
        cmap="tab10",
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_title("UMAP projection")

    # Plot t-SNE
    scatter2 = ax2.scatter(
        embedding_tsne[:, 0],
        embedding_tsne[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.6,
        s=100,
    )
    ax2.scatter(
        centroids_embedding_tsne[:, 0],
        centroids_embedding_tsne[:, 1],
        marker="*",
        s=300,
        c=range(len(cluster_centers)),
        cmap="tab10",
        edgecolor="black",
        linewidth=1.5,
    )
    ax2.set_title("t-SNE projection")

    # Add colorbar
    plt.colorbar(scatter1, ax=ax1, label="Cluster")
    plt.colorbar(scatter2, ax=ax2, label="Cluster")

    fig.suptitle(
        "Dimensionality reduction projections of reference features\n"
        "Points colored by cluster, centroids shown as stars",
        y=1.02,
    )

    return fig


def generate_combinations(n: int, k: int) -> list[list[int]]:
    """Generate all possible k-combinations from n elements.

    This function recursively generates all possible combinations of k elements
    chosen from a set of n elements (0 to n-1).

    Args:
        n: The total number of elements to choose from (0 to n-1)
        k: The number of elements to include in each combination

    Returns:
        list[list[int]]: A list of all possible k-combinations, where each combination
            is represented as a list of integers
    """
    if k > n:
        return []
    if k == 0:
        return [[]]
    if k == n:
        return [list(range(n))]
    res = []
    for i in range(n):
        res.extend([*j, i] for j in generate_combinations(i, k - 1))
    return res


class MaybeToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor."""

    def __call__(self, pic: PIL.Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.

        Args:
            pic: Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


def sample_points(
    points: np.ndarray,
    sample_range: tuple[int, int] = (4, 6),
    max_iterations: int = 30,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sample points by generating point sets of different sizes.

    Point sets can contain duplicates.
    Point sets have equal length so they can be batched. Note that each point sets has increased amount of
    points. e.g. subset0.shape = X, 1, 2 and subset1.shape = X, 2, 2, subset3.shape = X, 3, 2 etc.
    This is used to generate prompts with different granularity.

    For small point sets (≤8 points), generates all possible combinations.
    For larger sets (>8 points), uses random sampling to generate max_iterations samples.

    Args:
        points: Input points array of shape (N, 2) where N is number of points
        sample_range: Tuple of (min_points, max_points) to sample
        max_iterations: Maximum number of random sampling iterations for large point sets

    Returns:
        tuple containing:
            - sample_list: List of arrays, where each array has shape:
                - (max_iterations, i, 2) for large point sets (random sampling)
                - (n_combinations, i, 2) for small point sets (all combinations)
                where i ranges from sample_range[0] to sample_range[1], and 2 represents x,y coordinates
            - label_list: List of arrays, where each array has shape:
                - (max_iterations, i) for large point sets
                - (n_combinations, i) for small point sets
                containing ones for each sampled point

    """
    sample_list = []
    label_list = []
    for i in range(
        min(sample_range[0], len(points)),
        min(sample_range[1], len(points)) + 1,
    ):
        if len(points) > 8:
            index = [random.sample(range(len(points)), i) for j in range(max_iterations)] # nosec B311
            sample = np.take(points, index, axis=0)  # (max_iterations * i) * 2
        else:
            index = generate_combinations(len(points), i)
            sample = np.take(points, index, axis=0)  # i * n * 2

        # generate label  max_iterations * i
        label = np.ones((sample.shape[0], i))
        sample_list.append(sample)
        label_list.append(label)
    return sample_list, label_list


def masks_to_custom_masks(
    masks: list[torch.Tensor | None],
    class_id: int = 0,
) -> list[Masks]:
    """Converts torch masks to Masks objects.

    Args:
        masks: List of torch tensors with shape (N, H, W) containing masks,
                or None for samples without masks
        class_id: The class id to use for all masks

    Returns:
        List of Masks objects
    """
    mask_list = []
    for mask in masks:
        if mask is None:
            # Create empty Masks object for samples without masks
            mask_list.append(Masks())
        else:
            # mask_array has shape (N, H, W) - already binary masks per instance
            masks_obj = Masks()
            for instance_idx in range(mask.shape[0]):
                # Add each instance mask with the same class_id
                masks_obj.add(mask[instance_idx], class_id=class_id)
            mask_list.append(masks_obj)
    return mask_list
