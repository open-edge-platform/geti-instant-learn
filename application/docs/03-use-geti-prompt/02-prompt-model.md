# Prompt model

> This section is under development

Geti Prompt supports two prompting methods: visual and text. Depending on the model, you can use visual prompting, text prompting, or both simultaneously.

## Models

Geti Prompt is built upon its own VisionPrompt framework, which supports the latest state-of-the-art foundation models from the literature. These foundation models have been selected based on their benchmarking performance as well as their permissive licenses, so that they can be deployed by users without restrictions.

> TODO: include what pipelines are supported for visual prompt & text prompt, how pipelines are configured, where to find more info about the pipeline benchmarks and characteristics, etc. Finish by redirecting user to dedicated “VisionPrompt framework” section.

## Visual prompting

For visual prompting, users will be able to teach the model what it should be looking for by highlighting the object(s) of interest on an image. You can manually capture one or few reference images, define the label that correspond to the object that needs to be detected, and annotate the object accordingly to create a prompt. Based on this prompt, the model will then infer on the input data to try to detect the object of interest.

### Capture image

> TODO: include how users can capture reference image and why, short gif

### Labels

> TODO: include how users can create label(s) and why, short gif

### Annotation

> TODO: include how users can annotate with the tools that are available, how to use other options in the Prompt canvas, what if the desired object cannot be selected, how to save prompt, what happens when saving prompt, etc – short gif

### Manage prompts

> TODO: include how to modify prompts, add new labels, remove prompts, and what will happen when modifying/deleting prompts, etc

## Text prompting

For text prompting, users will be able to teach the model what it should be looking for by simply providing a query in the UI.

> TODO: include how to prompt by text, what the difference is with visual prompting, how to manage prompts, etc – short gif
