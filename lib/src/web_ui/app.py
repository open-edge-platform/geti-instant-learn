# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This is the main file for the web UI.

It is a Flask application that allows you to run several Visual Prompting pipelines and see the results.
The web UI is served at http://127.0.0.1:5050

The web UI can be started by running:
python -m web_ui.app
"""

import argparse
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from getiprompt.utils.args import get_arguments
from getiprompt.utils.constants import DatasetName, PipelineName, SAMModelName
from getiprompt.utils.data import load_dataset
from web_ui.helpers import (
    load_and_prepare_data,
    parse_request_and_check_reload,
    prepare_reference_data,
    reload_pipeline_if_needed,
    stream_inference,
)

BATCH_SIZE = 5

warnings.filterwarnings("ignore", category=UserWarning)


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
app = Flask(__name__, static_folder="static", template_folder="templates")
initial_default_args = get_arguments([])
current_pipeline_args = argparse.Namespace(**vars(initial_default_args))

app.logger.info("Deferring pipeline loading until first request.")
current_pipeline_instance = None
current_pipeline_name = initial_default_args.pipeline


@app.route("/")
def index() -> str:
    """Serves the main HTML page."""
    ui_pipelines = [p.value for p in PipelineName if p != PipelineName.GROUNDING_DINO_SAM]
    ui_datasets = [d.value for d in DatasetName]
    return render_template(
        "index.html",
        sam_names=[model.value for model in SAMModelName],
        pipelines=ui_pipelines,
        datasets=ui_datasets,
        compile_models=initial_default_args.compile_models,
        default_sam_name=initial_default_args.sam,
        precision=initial_default_args.precision,
    )


@app.route("/api/classes")
def get_classes() -> tuple[Response, int]:
    """Returns a list of unique class names for a given dataset."""
    dataset_name = request.args.get("dataset", "PerSeg")
    try:
        full_dataset = load_dataset(dataset_name)
        unique_classes = full_dataset.get_categories()
        return jsonify({"classes": unique_classes})
    except FileNotFoundError:
        app.logger.error(f"Dataset '{dataset_name}' files not found.", exc_info=True)
        return jsonify({"error": f"Dataset '{dataset_name}' files not found."}), 404
    except Exception as e:
        app.logger.error(
            f"Error getting classes for {dataset_name}: {e}",
            exc_info=True,
        )
        return jsonify({"error": "Could not retrieve class list."}), 500


@app.route("/api/class_info")
def get_class_info() -> tuple[Response, int]:
    """Returns the total number of images for a given class in a dataset."""
    dataset_name = request.args.get("dataset")
    class_name = request.args.get("class_name")

    if not dataset_name or not class_name:
        return jsonify({"error": "Missing dataset or class_name parameter"}), 400

    try:
        full_dataset = load_dataset(dataset_name)
        # Assuming dataset object has a method to get image count per class
        count = full_dataset.get_instance_count_per_category(class_name)
        return jsonify({"total_images": count})

    except Exception as e:
        app.logger.error(
            f"Error getting image count for {class_name} in {dataset_name}: {e}",
            exc_info=True,
        )
        return jsonify({"error": "Could not retrieve image count."}), 500


@app.route("/api/process", methods=["POST"])
def run_processing() -> Response:
    """Main endpoint to run the visual prompting pipeline and stream results.

    Handles request parsing, pipeline reloading, data preparation, learning,
    and streaming inference results.
    """
    global current_pipeline_args, current_pipeline_instance, current_pipeline_name

    try:
        request_data = request.json
        reload_needed, requested_values, parsed_args = parse_request_and_check_reload(
            request_data,
            current_pipeline_name,
            current_pipeline_args,
        )

        current_pipeline_instance, current_pipeline_name, current_pipeline_args = reload_pipeline_if_needed(
            reload_needed,
            requested_values,
            parsed_args,
            current_pipeline_instance,
        )

        if current_pipeline_instance is None:
            msg = f"Pipeline instance for '{current_pipeline_name}' is not loaded."
            raise ValueError(msg)  # noqa: TRY301

        selected_pipeline = current_pipeline_instance
        dataset_name = request_data.get("dataset", "PerSeg")
        class_name_filter = request_data.get("class_name", "can")
        n_shot = int(request_data.get("n_shot", 1))
        random_prior = request_data.get("random_prior", False)
        num_target_images = request_data.get("num_target_images")

        (
            reference_images,
            reference_priors,
            target_indices,
            full_dataset,
        ) = load_and_prepare_data(
            dataset_name,
            class_name_filter,
            n_shot,
            num_target_images,
            random_prior=random_prior,
        )

        selected_pipeline.learn(reference_images=reference_images, reference_priors=reference_priors)

        prepared_reference_data = prepare_reference_data(reference_images, reference_priors)

        return Response(
            stream_with_context(
                stream_inference(
                    pipeline=selected_pipeline,
                    full_dataset=full_dataset,
                    target_indices=target_indices,
                    class_name_filter=class_name_filter,
                    prepared_reference_data=prepared_reference_data,
                    batch_size=BATCH_SIZE,
                )
            ),
            mimetype="application/json",
        )

    except (ValueError, FileNotFoundError, KeyError) as e:
        app.logger.error(f"Client or data error in run_processing: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in run_processing: {e}", exc_info=True)
        return (
            jsonify({"error": f"An unexpected server error occurred: {type(e).__name__}"}),
            500,
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050)
