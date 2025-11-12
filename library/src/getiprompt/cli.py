# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This module contains the Geti Prompt CLI."""

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from getiprompt.models import Model
from getiprompt.scripts.benchmark import perform_benchmark_experiment
from getiprompt.scripts.run import run_model
from getiprompt.utils.args import populate_benchmark_parser
from getiprompt.utils.utils import setup_logger

setup_logger()


class GetiPromptCLI:
    """This class is the entry point for the Geti Prompt CLI."""

    def __init__(self) -> None:
        """Initialize the Geti Prompt CLI."""
        self.parser = ArgumentParser(description="Geti Prompt CLI", env_prefix="getiprompt")
        self._add_subcommands()
        self.execute()

    @staticmethod
    def add_run_arguments(parser: ArgumentParser) -> None:
        """Add arguments for the run subcommand."""
        parser.add_subclass_arguments(Model, "model", required=True)
        parser.add_argument(
            "--data_root",
            "--data",
            type=str,
            default=None,
            help=(
                "Root directory containing the dataset using FolderDataset structure. "
                "Required: root/images/{category}/*.jpg and root/masks/{category}/*.png. "
                "Each category must have at least (n_shots + 1) images and masks. "
                "Example: ~/data/dataset/images/apple/1.jpg with ~/data/dataset/masks/apple/1.png"
            ),
        )
        parser.add_argument(
            "--text_prompt",
            type=str,
            default=None,
            help=(
                "Text prompt with comma/dot-separated categories. If provided, no images are needed. "
                "Model must be explicitly set (e.g., use GroundedSAM for text prompts)."
            ),
        )
        parser.add_argument(
            "--n_shots",
            type=int,
            default=1,
            help=(
                "Number of reference shots per category. Each category must have at least"
                "(n_shots + 1) images and masks. Defaults to 1."
            ),
        )
        parser.add_argument("--output_location", type=str, default=None, help="Directory to save output.")
        parser.add_argument(
            "--output_masks_only",
            action="store_true",
            default=False,
            help="Whether to save masks only.",
        )
        parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing target images.")

    @staticmethod
    def add_benchmark_arguments(parser: ArgumentParser) -> None:
        """Add arguments for the benchmark subcommand."""
        # TODO(Daankrol): rewrite benchmark script into a class and add arguments here  # noqa: TD003
        populate_benchmark_parser(parser)

    def execute(self) -> None:
        """Execute the CLI."""
        cfg = self.parser.parse_args()
        log_level = cfg[cfg.subcommand].log_level if "log_level" in cfg[cfg.subcommand] else "INFO"
        setup_logger(log_level=log_level)
        instantiated_config = self.parser.instantiate_classes(cfg)

        self._execute_subcommands(instantiated_config)

    @staticmethod
    def _getiprompt_subcommands() -> dict[str, str]:
        """Returns the subcommands and help messages for each subcommand."""
        return {
            "run": "Perform both learning and inference steps.",
            "benchmark": "Run benchmarking on the models.",
        }

    def _add_subcommands(self) -> None:
        """Registers the subcommands for the CLI."""
        parser_subcommands = self.parser.add_subcommands()

        for name, description in self._getiprompt_subcommands().items():
            parser = ArgumentParser(description=description)
            self._add_common_args(parser)
            getattr(self, f"add_{name}_arguments")(parser)
            parser_subcommands.add_subcommand(name, parser)

    @staticmethod
    def _add_common_args(parser: ArgumentParser) -> None:
        """Adds common arguments for all subcommands."""
        parser.add_argument("--config", action=ActionConfigFile)
        parser.add_argument(
            "--log_level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set the logging level.",
        )

    @staticmethod
    def _execute_subcommands(config: Namespace) -> None:
        """Execute the appropriate subcommand based on the config.

        Args:
            config: The configuration namespace.

        Raises:
            ValueError: If the subcommand is invalid.
        """
        subcommand = config.subcommand
        match subcommand:
            case "run":
                model = config.run.model
                run_model(
                    model=model,
                    data_root=config.run.data_root,
                    text_prompt=config.run.text_prompt,
                    output_location=config.run.output_location,
                    batch_size=config.run.batch_size,
                    n_shots=config.run.n_shots,
                )
            case "benchmark":
                perform_benchmark_experiment(config.benchmark)
            case _:
                msg = f"Invalid subcommand: {subcommand}"
                raise ValueError(msg)


def main() -> None:
    """Main function for the CLI."""
    GetiPromptCLI()


if __name__ == "__main__":
    main()
