# Geti Prompt Development UI

An interactive web-based development interface for visualizing and debugging visual prompting pipelines.

## Overview

The Development UI is a Flask application that provides a visual interface for experimenting with different visual prompting pipelines. It allows you to:

- Select and configure different pipelines (Matcher, SoftMatcher, PerDino, etc.)
- Choose from available datasets (LVIS, PerSeg)
- Visualize similarity maps, generated masks, and point prompts
- Compare predicted masks with ground truth masks
- Adjust model parameters in real-time

## Prerequisites

Before running the Development UI, you must have the `getiprompt` library installed with the `dev_ui` optional dependencies. The UI depends on the library's models, components, and utilities, plus Flask for the web server.

### Installation

From the root of the `library` directory:

```bash
# Install the library with dev_ui dependencies
uv sync --extra dev_ui

# Or install with all dependencies (includes dev_ui)
uv sync --extra full

# Or if using pip
pip install -e ".[dev_ui]"
```

## Running the Development UI

Navigate to the `dev_ui` directory and run the application:

```bash
cd library/dev_ui
python app.py
```

The UI will start on `http://127.0.0.1:5050` by default.

### Alternative: Using Flask CLI

You can also run the application using Flask's development server:

```bash
cd library/dev_ui
python -m flask run --host=127.0.0.1 --port=5050
```

### Configuration

You can modify the host, port, and debug settings by editing the values at the bottom of `app.py`:

```python
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
```

Or set Flask environment variables:

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_RUN_HOST=0.0.0.0
export FLASK_RUN_PORT=8080
flask run
```

## Usage

1. **Select a Pipeline**: Choose from available visual prompting algorithms (Matcher, SoftMatcher, PerDino, etc.)

2. **Choose a Dataset**: Select a dataset to load images from (e.g., LVIS, PerSeg)

3. **Select a Class**: Pick a specific object class to work with (e.g., "can", "cat", "chair")

4. **Configure Parameters**:
   - SAM Model: Select the SAM backbone (SAM-HQ-tiny, MobileSAM, etc.)
   - Encoder Model: Choose the feature encoder (dinov3_large, etc.)
   - N-shot: Number of reference examples
   - Precision: Model precision (bf16, fp16, fp32)
   - Other pipeline-specific parameters

5. **Run Inference**: Click "Process" to run the pipeline on target images

6. **Visualize Results**:
   - Reference images and masks (green)
   - Target images with predicted masks (red)
   - Ground truth masks (green overlay)
   - Similarity maps (heatmaps)
   - Point prompts used for segmentation

## Features

### Real-time Model Switching

The UI intelligently detects when critical parameters change (pipeline, SAM model, encoder) and automatically reloads the model while preserving non-critical parameter changes.

### Streaming Results

Inference results are streamed incrementally, allowing you to see outputs as they're generated rather than waiting for all images to complete.

### Interactive Visualization

- Toggle between different visualization layers
- Inspect individual similarity maps
- View point prompts overlaid on images
- Compare predictions with ground truth

## Technical Details

### Architecture

The Development UI is built with:

- **Backend**: Flask (Python web framework)
- **Frontend**: Vanilla JavaScript with HTML/CSS
- **Data Format**: Server-sent events for streaming results
- **Image Encoding**: Base64-encoded PNG data URIs

### File Structure

```
dev_ui/
├── app.py              # Main Flask application
├── helpers.py          # Helper functions for data processing
├── static/
│   └── script.js       # Frontend JavaScript code
├── templates/
│   └── index.html      # Main HTML template
└── README.md           # This file
```

### Dependencies

The Development UI requires the following from the `getiprompt` library:

- `getiprompt.models`: Pipeline models
- `getiprompt.data`: Dataset loaders
- `getiprompt.utils`: Utility functions and constants
- `getiprompt.components`: Encoders and other components
- `getiprompt.types`: Type definitions

Additionally requires:
- Flask
- OpenCV (cv2)
- NumPy
- PyTorch

## Troubleshooting

### Port Already in Use

If port 5050 is already in use, modify the port in `app.py` or use Flask environment variables.

### Model Loading Errors

Ensure the `getiprompt` library is properly installed and all model weights are available. Check the console output for detailed error messages.

### Dataset Not Found

Make sure the required datasets are downloaded and accessible. Dataset paths are typically configured in the library's data loading utilities.

### Memory Issues

If you encounter out-of-memory errors:
- Use smaller SAM models (MobileSAM, SAM-HQ-tiny)
- Reduce the number of target images processed
- Lower the batch size in `app.py` (default is 5)

## Development

To modify the UI:

1. **Backend Changes**: Edit `app.py` or `helpers.py` for Flask routes and data processing
2. **Frontend Changes**: Edit `templates/index.html` or `static/script.js` for UI and interactions
3. **Styling**: Add CSS to the `<style>` section in `index.html`

The Flask development server automatically reloads when you save changes to Python files.

## License

This Development UI is part of the Geti Prompt project and is licensed under the Apache 2.0 License.
