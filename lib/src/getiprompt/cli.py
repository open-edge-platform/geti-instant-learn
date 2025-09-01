# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This module contains the Geti Prompt CLI."""

import inspect
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from getiprompt.benchmark import perform_benchmark_experiment
from getiprompt.pipelines import GroundingDinoSAM
from getiprompt.pipelines.pipeline_base import Pipeline
from getiprompt.run import run_pipeline
from getiprompt.utils.args import populate_benchmark_parser


class GetiPromptCLI:
    """This class is the entry point for the Geti Prompt CLI."""

    def __init__(self) -> None:
        self.parser = ArgumentParser(description="Geti Prompt CLI", env_prefix="getiprompt")
        self._add_subcommands()
        self.execute()

    @staticmethod
    def add_run_arguments(parser: ArgumentParser) -> None:
        """Add arguments for the run subcommand."""
        # load datasets
        parser.add_subclass_arguments(Pipeline, "pipeline", default="Matcher")
        parser.add_argument(
            "--reference_images", "--ref", type=str, default=None, help="Directory with reference images."
        )
        parser.add_argument(
            "--target_images", "--target", type=str, required=True, help="Directory with target images."
        )
        parser.add_argument(
            "--reference_prompts",
            "--ref_prompt",
            type=str,
            default=None,
            help="Directory with reference prompts (masks or points).",
        )
        parser.add_argument(
            "--points", type=str, default=None, help="Reference points as a string. e.g. [0:[640,640], -1:[200,200]]"
        )
        parser.add_argument(
            "--reference_text_prompt",
            "--text",
            type=str,
            default=None,
            help="Text prompt for grounding dino. If provided, pipeline is set to GroundingDinoSAM.",
        )
        parser.add_argument("--output_location", type=str, default=None, help="Directory to save output.")
        parser.add_argument(
            "--output_masks_only", action="store_true", default=False, help="Whether to save masks only."
        )
        parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing target images.")

    @staticmethod
    def add_benchmark_arguments(parser: ArgumentParser) -> None:
        """Add arguments for the benchmark subcommand."""
        # TODO(Daankrol): rewrite benchmark script into a class and add arguments here  # noqa: TD003
        populate_benchmark_parser(parser)

    @staticmethod
    def add_ui_arguments(parser: ArgumentParser) -> None:
        """Add arguments for the ui subcommand."""
        parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the UI on.")
        parser.add_argument("--debug", type=bool, default=True, help="Whether to run the UI in debug mode.")
        parser.add_argument("--port", type=int, default=5050, help="Port to run the UI on.")

    @staticmethod
    def _set_text_prompt_args(cfg: Namespace) -> None:
        """Switch to GroundingDinoSAM pipeline if text prompt is provided."""
        if cfg.subcommand == "run" and cfg.run.reference_text_prompt:
            sig = inspect.signature(GroundingDinoSAM.__init__)
            gino_accepted_args = {p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD}

            cfg.run.pipeline.init_args = {
                key: value for key, value in cfg.run.pipeline.init_args.items() if key in gino_accepted_args
            }
            cfg.run.pipeline.class_path = "getiprompt.pipelines.GroundingDinoSAM"

    def execute(self) -> None:
        """Execute the CLI."""
        cfg = self.parser.parse_args()
        self._set_text_prompt_args(cfg)

        instantiated_config = self.parser.instantiate_classes(cfg)

        self._execute_subcommands(instantiated_config)

    @staticmethod
    def _getiprompt_subcommands() -> dict[str, str]:
        """Returns the subcommands and help messages for each subcommand."""
        return {
            "run": "Perform both learning and inference steps.",
            "benchmark": "Run benchmarking on the pipelines.",
            "ui": "Run the UI for the pipelines.",
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

    @staticmethod
    def _execute_subcommands(config: Namespace) -> None:
        """Run the appropriate subcommand based on the config."""
        subcommand = config.subcommand
        match subcommand:
            case "run":
                if not config.run.reference_images and not config.run.reference_text_prompt:
                    msg = "Either reference_images or reference_text_prompt must be provided."
                    raise ValueError(msg)

                pipeline = config.run.pipeline
                run_pipeline(
                    pipeline=pipeline,
                    target_images=config.run.target_images,
                    reference_images=config.run.reference_images,
                    reference_prompts=config.run.reference_prompts,
                    reference_points_str=config.run.points,
                    reference_text_prompt=config.run.reference_text_prompt,
                    output_location=config.run.output_location,
                    output_masks_only=config.run.output_masks_only,
                    batch_size=config.run.batch_size,
                )
            case "benchmark":
                perform_benchmark_experiment(config.benchmark)
            case "ui":
                from web_ui.app import app

                app.run(host=config.ui.host, debug=config.ui.debug, port=config.ui.port)
            case _:
                msg = f"Invalid subcommand: {subcommand}"
                raise ValueError(msg)


def main() -> None:
    """Main function for the CLI."""
    GetiPromptCLI()


if __name__ == "__main__":
    main()
