# Concepts & Developer Guide

Technical background and architectural concepts to help developers and advanced users understand Geti Prompt Application.

## Table of Contents

### Core Concepts
<!-- TODO: Explain key terms like Visual Prompting, Zero-shot Learning, Few-shot Learning -->
- **Visual Prompting**: How user interactions translate to model inputs.
- **Pipelines**: The flow of data from source to inference to sink.
- **Sources & Sinks**: Abstractions for input and output data streams.

### System Architecture
<!-- TODO: Diagram and explanation of the Frontend-Backend-Model interaction -->
- **Frontend (React)**: UI components and state management.
- **Backend (FastAPI)**: API structure, WebSocket handling, and task management.
- **Inference Engine**: How models are loaded, optimized (OpenVINO), and executed.

### Integration & Extensibility
<!-- TODO: Guide on how to extend the platform -->
- **Custom Drivers**: Adding support for new camera types.
- **Model Registry**: Integrating new foundation models.
- **API Integration**: Consuming inference results via REST or MQTT.

### Database & Data Storage
<!-- TODO: Explain data persistence strategy -->
- **Prompt Storage**: How visual and text prompts are saved and versioned.
- **Configuration**: Managing application settings and pipeline configs.

## Related Documentation

- **[Tutorials](../tutorials/01-tutorials.md)**: Step-by-step guides for common tasks
- **[How-To Guides](../how-to-guides/01-how-to-guids.md)**: Task-focused guides for specific features
- **[Architecture Guide](02-development.md)**: Original architecture documentation
- **[Geti Prompt Library](../../../library/docs/01-introduction.md)**: Reusable components and algorithms

## Discussion & Questions

For conceptual discussions or questions:

- Open an issue on GitHub with label "discussion"
- Refer to relevant concept document in your issue
- Share findings and updates in issue comments
