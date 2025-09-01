# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for all pipelines."""

from abc import ABC, abstractmethod
from logging import getLogger

from getiprompt.processes.preprocessors import ResizeImages, ResizeMasks
from getiprompt.processes.process_base import Process
from getiprompt.types import Image, Priors, Results

logger = getLogger("Vision Prompt")


class Pipeline(ABC):
    """This class is the base class for all pipelines.

    Examples:
        >>> import numpy as np
        >>> from getiprompt.pipelines import Pipeline
        >>> from getiprompt.types import Image, Priors, Results
        >>>
        >>> class MyPipeline(Pipeline):
        ...     def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        ...         self.resize_masks(reference_priors)
        ...         return Results()
        ...     def infer(self, target_images: list[Image]) -> Results:
        ...         self.resize_images(images=target_images)
        ...         return Results()
        >>>
        >>> my_pipeline = MyPipeline(image_size=512)
        >>> sample_image = np.zeros((10, 10, 3), dtype=np.uint8)
        >>> learn_results = my_pipeline.learn([Image(sample_image)], [Priors()])
        >>> infer_results = my_pipeline.infer([Image(sample_image)])
        >>>
        >>> isinstance(learn_results, Results) and isinstance(infer_results, Results)
        True
    """

    def __init__(
        self,
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        """Initialization method that caches all parameters.

        Args:
            image_size: The size of the image to use, if None, the image will not be resized.
        """
        self.resize_images = ResizeImages(size=image_size)
        self.resize_masks = ResizeMasks(size=image_size)

    @abstractmethod
    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> None:
        """This method learns the context.

        Args:
            reference_images: A list of images ot learn from.
            reference_priors: A list of priors associated with the image.

        """

    @abstractmethod
    def infer(self, target_images: list[Image]) -> Results:
        """This method uses the learned context to infer object locations.

        Args:
            target_images: A List of images to infer.

        Returns:
            None
        """

    def _get_process_durations(self) -> list[tuple[str, float]]:
        """Get the durations of the processes.

        Returns:
            A list of tuples containing the name of the component and the duration.
        """
        return [
            (attr_value.__class__.__name__, attr_value.last_duration)
            for attr_value in self.__dict__.values()
            if isinstance(attr_value, Process) and hasattr(attr_value, "last_duration")
        ]

    def _reset_process_durations(self) -> None:
        """Reset the durations of the processes."""
        for attr_value in self.__dict__.values():
            if isinstance(attr_value, Process):
                attr_value.last_duration = 0.0

    def log_timing(self, title: str = "Inference") -> float:
        """Print the timing of the processes in a table.

        Args:
            title: The title of the table.

        Returns:
            The total time of the processes.
        """
        process_durations = self._get_process_durations()
        output_str = f"\n--- {title} Timings ---"
        max_name_len = max((len(name) for name, _ in process_durations), default=0)
        max_name_len = max(max_name_len, len("Total"))

        total_time = sum(t for _, t in process_durations)
        output_str += f"\n{'Name':<{max_name_len}} | {'Total':<10}"
        output_str += "\n" + "-" * (max_name_len + 10 + 20 + 3)
        for name, duration in process_durations:
            output_str += f"\n{name:<{max_name_len}} | {duration:<10.4f}"
        output_str += "\n" + "-" * (max_name_len + 10 + 20 + 3)
        output_str += f"\n{'Total':<{max_name_len}} | {total_time:<10.4f}"
        output_str += "\n" + "-" * (max_name_len + 10 + 20 + 3) + "\n"
        logger.debug(output_str)
        return total_time
