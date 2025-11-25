# Geti Prompt

Geti Prompt is a modular framework for few-shot visual segmentation using visual prompting techniques. Enables easy experimentation with different algorithms, backbones (SAM, MobileSAM, EfficientViT-SAM, DinoV2), and project components for finding and segmenting objects from just a few examples.

# Base URL


| URL | Description |
|-----|-------------|


# APIs

## GET /health

Health Check

Health check endpoint




### Responses

#### 200


Successful Response


object







## POST /api/v1/projects/{project_id}/frames

Capture Frame

Capture the latest frame from the video stream of the active project.
Returns the frame ID in the response body and a Location header pointing to the captured frame.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Responses

#### 201


Frame captured successfully


object


| Field | Type | Description |
|-------|------|-------------|
| frame_id | string |  |





#### 404


Project not found or no connected source







Examples





```json
{
  "detail": "Resource PROJECT with id 123e4567-e89b-12d3-a456-426614174000 not found"
}
```




```json
{
  "detail": "Project 123e4567-e89b-12d3-a456-426614174000 has no connected source. Please connect a source before capturing frames."
}
```



#### 400


Project is not active or frame capture failed







Examples





```json
{
  "detail": "Cannot capture frame: project 123e4567-e89b-12d3-a456-426614174000 is not active. Please activate the project before capturing frames."
}
```




```json
{
  "detail": "No frame received within 5.0 seconds. Pipeline may not be running."
}
```




```json
{
  "detail": "Frame capture failed: internal processing error"
}
```



#### 500


Internal server error








#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}/frames/{frame_id}

Get Frame

Retrieve a captured frame as JPEG.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| frame_id | string | True |  |


### Responses

#### 200


Frame retrieved successfully











Examples





```json
"FFD8FFE000104A46494600010100000100010000FFDB..."
```



#### 404


Frame or project not found








#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /api/v1/projects/{project_id}/labels

Create Label

Create a new label with the given name.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Request Body

[LabelCreateSchema](#labelcreateschema)







### Responses

#### 200


Successful Response








#### 201


Successfully created a new label.




#### 404


Project not found.




#### 409


Label with this name already exists.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}/labels

Get All Labels

Get all labels for selected project


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| offset | integer | False |  |
| limit | integer | False |  |


### Responses

#### 200


Successfully retrieved a list of all labels for selected project.


[LabelsListSchema](#labelslistschema)







#### 404


Project not found.




#### 500


Unexpected error occurred while retrieving available project configurations.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}/labels/{label_id}

Get Label By Id

Get a label by its ID for selected project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| label_id | string | True |  |


### Responses

#### 200


Successfully retrieved the details of label.


[LabelSchema](#labelschema)







#### 404


Project or label not found.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## DELETE /api/v1/projects/{project_id}/labels/{label_id}

Delete Label By Id

Delete a label by its ID for selected project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| label_id | string | True |  |


### Responses

#### 204


Successfully deleted the label.




#### 404


Project or label not found.




#### 500


Unexpected error occurred while deleting the label.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## PUT /api/v1/projects/{project_id}/labels/{label_id}

Update Label

Update the label.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| label_id | string | True |  |


### Request Body

[LabelUpdateSchema](#labelupdateschema)







### Responses

#### 200


Successfully updated the label.


[LabelSchema](#labelschema)







#### 404


Project or label not found.




#### 409


Label name already exists.




#### 500


Unexpected error occurred while updating the label.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}/models

Get All Models

Retrieve the all model configurations of the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| offset | integer | False |  |
| limit | integer | False |  |


### Responses

#### 200


Successfully retrieved the active model configuration for the project.


[ProcessorListSchema](#processorlistschema)







#### 404


Project not found







Examples





```json
{
  "detail": "Project with ID 123e4567-e89b-12d3-a456-426614174000 not found."
}
```



#### 500


Unexpected error occurred








#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /api/v1/projects/{project_id}/models

Create Model

Create a new model configuration for the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Request Body

[ProcessorCreateSchema](#processorcreateschema)







### Responses

#### 201


Model configuration created successfully








#### 404


Project not found








#### 400


Invalid model configuration data








#### 409


Conflicting data was provided








#### 500


Unexpected error occurred








#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}/models/active

Get Active Model

Retrieve the active model configuration of the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Responses

#### 200


Successfully retrieved the active model configuration for the project.


[ProcessorSchema](#processorschema)







#### 404


Project or active model configuration not found







Examples





```json
{
  "detail": "Project with ID 123e4567-e89b-12d3-a456-426614174000 not found."
}
```




```json
{
  "detail": "No active model configuration found for specified project"
}
```



#### 500


Unexpected error occurred








#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}/models/{model_id}

Get Model

Retrieve the model configuration of the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| model_id | string | True |  |


### Responses

#### 200


Successfully retrieved the model configuration for the project.


[ProcessorSchema](#processorschema)







#### 404


Project or model configuration not found







Examples





```json
{
  "detail": "Project with id 123e4567-e89b-12d3-a456-426614174000 not found."
}
```




```json
{
  "detail": "No active model configuration found for the specified project."
}
```



#### 500


Unexpected error occurred








#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## PUT /api/v1/projects/{project_id}/models/{model_id}

Update Model

Update an existing model configuration for the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| model_id | string | True |  |


### Request Body

[ProcessorUpdateSchema](#processorupdateschema)







### Responses

#### 200


Model configuration updated successfully


[ProcessorSchema](#processorschema)







#### 404


Project or model configuration not found







Examples





```json
{
  "detail": "Project with ID 123e4567-e89b-12d3-a456-426614174000 not found."
}
```




```json
{
  "detail": "No active model configuration found for the specified project."
}
```



#### 400


Invalid update data








#### 500


Unexpected error occurred








#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## DELETE /api/v1/projects/{project_id}/models/{model_id}

Delete Model

Delete a model configuration from the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| model_id | string | True |  |


### Responses

#### 204


Model configuration deleted successfully




#### 404


Project or model configuration not found







Examples





```json
{
  "detail": "Project with ID 3fa85f64-5717-4562-b3fc-2c963f66afa6 not found."
}
```




```json
{
  "detail": "Processor with ID 04b34cb0-c405-4566-990a-4eaeeaaa515a not found."
}
```



#### 500


Unexpected error occurred








#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /api/v1/projects

Create Project

Create a new project with the given name.




### Request Body

[ProjectCreateSchema](#projectcreateschema)







### Responses

#### 201


Successfully created a new project.








#### 409


Project with this name already exists.




#### 500


Unexpected error occurred while creating a new project.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects

Get Projects List

Retrieve a list of all available project configurations.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| offset | integer | False |  |
| limit | integer | False |  |


### Responses

#### 200


Successfully retrieved a list of all available project configurations.


[ProjectsListSchema](#projectslistschema)







#### 500


Unexpected error occurred while retrieving available project configurations.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## DELETE /api/v1/projects/{project_id}

Delete Project

Delete the specified project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Responses

#### 204


Successfully deleted the project.




#### 500


Unexpected error occurred while deleting the project.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}

Get Project

Retrieve the project's configuration.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Responses

#### 200


Successfully retrieved the configuration for a project.


[ProjectSchema](#projectschema)







#### 404


Project not found.




#### 500


Unexpected error occurred while retrieving the configuration of a project.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## PUT /api/v1/projects/{project_id}

Update Project

Update the project's configuration.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Request Body

[ProjectUpdateSchema](#projectupdateschema)







### Responses

#### 200


Successfully updated the configuration for the project.


[ProjectSchema](#projectschema)







#### 404


Project not found.




#### 409


Project name already exists.




#### 500


Unexpected error occurred while updating the configuration of the project.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/active

Get Active Project

Retrieve the configuration of the currently active project.




### Responses

#### 200


Successfully retrieved the configuration of the currently active project.


[ProjectSchema](#projectschema)







#### 404


No active project found.




#### 500


Unexpected error occurred while retrieving the active project configuration.




## GET /api/v1/projects/export

Export Projects

Export project configurations as a zip archive.
If no names are provided, exports all projects.

Returns:
    Response: A .zip file containing the selected project directories (e.g., {p1_name}/configuration.yaml).


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| names |  | False |  |


### Responses

#### 200


Successfully exported the project configurations as a zip archive.








#### 500


Unexpected error occurred while exporting the project configurations.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /api/v1/projects/import

Import Projects

Import projects from a .zip archive.
The server will copy the project configurations into the application's configuration directory.
If a project with the same name already exists, the import for that specific project
will be rejected with an error to prevent accidental overwrites.




### Responses

#### 201


Successfully imported a new project from an archive.








#### 500


Unexpected error occurred while importing the project.




## GET /api/v1/projects/{project_id}/prompts

Get All Prompts

Retrieve a list of all prompts for the project with pagination.
Visual prompts include thumbnail previews in the list response.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| offset | integer | False |  |
| limit | integer | False |  |


### Responses

#### 200


Successfully retrieved the list of all prompts for the project. Visual prompts include thumbnails.


[PromptsListSchema](#promptslistschema)







#### 404


Project not found.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /api/v1/projects/{project_id}/prompts

Create Prompt

Create a new text or visual prompt for the project.
Text prompts are limited to one per project.
Visual prompts must reference an existing frame and include annotations.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Request Body









### Responses

#### 201


Successfully created a new prompt.










#### 404


Project, label, or frame not found.




#### 409


Prompt already exists.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}/prompts/{prompt_id}

Get Prompt

Retrieve the details of a specific prompt.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| prompt_id | string | True |  |


### Responses

#### 200


Successfully retrieved the details of the prompt.










#### 404


Project or prompt not found.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## PUT /api/v1/projects/{project_id}/prompts/{prompt_id}

Update Prompt

Update an existing prompt (text or visual) for the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| prompt_id | string | True |  |


### Request Body









### Responses

#### 200


Successfully updated the prompt.










#### 404


Project, prompt, label, or frame not found.




#### 400


Invalid update data or type mismatch.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## DELETE /api/v1/projects/{project_id}/prompts/{prompt_id}

Delete Prompt

Delete a prompt from the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| prompt_id | string | True |  |


### Responses

#### 204


Successfully deleted the prompt.




#### 404


Project or prompt not found.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}/sink

Get Sink

Retrieve the sink configuration of the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Responses

#### 200


Successfully retrieved the sink configuration for the project.








#### 500


Unexpected error occurred while retrieving the sink configuration of the project.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## PUT /api/v1/projects/{project_id}/sink

Update Sink

Update the project's configuration.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Responses

#### 200


Successfully updates the configuration for the project's sink.








#### 201


Successfully created the configuration for the project's sink.




#### 500


Unexpected error occurred while updating the configuration of the project's sink.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## DELETE /api/v1/projects/{project_id}/sink

Delete Sink

Delete the specified project's sink configuration.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Responses

#### 200


Successfully deleted the project's sink configuration.








#### 500


Unexpected error occurred while deleting the project's sink configuration.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}/sources

Get Sources

Retrieve the source configuration of the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Responses

#### 200


Successfully retrieved the sources configuration for the project.


[SourcesListSchema](#sourceslistschema)







#### 404


Project not found.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /api/v1/projects/{project_id}/sources

Create Source

Create a new source configuration for the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Request Body

[SourceCreateSchema](#sourcecreateschema)







### Responses

#### 201


Source created.


[SourceSchema](#sourceschema)







#### 404


Project not found.




#### 409


Source of this type already exists in project.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## PUT /api/v1/projects/{project_id}/sources/{source_id}

Update Source

Update the project's source configuration.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| source_id | string | True |  |


### Request Body

[SourceUpdateSchema](#sourceupdateschema)







### Responses

#### 200


Successfully updated the configuration for the project's source.


[SourceSchema](#sourceschema)







#### 404


Project or source not found.




#### 409


Source type change is not allowed.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## DELETE /api/v1/projects/{project_id}/sources/{source_id}

Delete Source

Delete the specified project's source configuration.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| source_id | string | True |  |


### Responses

#### 204


Successfully deleted the project's source configuration.




#### 500


Unexpected error occurred while deleting the project's source configuration.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}/sources/{source_id}/frames

Get Frames

Retrieve a paginated list of frames from the source.
Only available for seekable sources (e.g., image folders, video files).
The source must be the currently connected source in the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| source_id | string | True |  |
| offset | integer | False |  |
| limit | integer | False |  |


### Responses

#### 200


Successfully retrieved the list of frames for the project's source.


[FrameListResponse](#framelistresponse)







#### 400


Source does not support frame navigation or source is not connected.




#### 404


Project or source not found.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/v1/projects/{project_id}/sources/{source_id}/frames/index

Get Frame Index

Get the current frame index from the source.
Only available for seekable sources (e.g., image folders, video files).
The source must be the currently connected source in the project.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| source_id | string | True |  |


### Responses

#### 200


Successfully retrieved the current frame index.


[FrameIndexResponse](#frameindexresponse)







#### 400


Source does not support frame navigation or source is not connected.




#### 404


Project or source not found.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /api/v1/projects/{project_id}/sources/{source_id}/frames/{index}

Seek Frame

Seek to a specific frame in the source.
Only available for seekable sources (e.g., image folders, video files).
The source must be the currently connected source in the project.

The UI can use this for "Next", "Prev", "First", "Last" navigation:
- First: index = 0
- Last: Get total from list_frames, then seek to total - 1
- Next: Get current index, then seek to index + 1
- Prev: Get current index, then seek to index - 1


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |
| source_id | string | True |  |
| index | integer | True |  |


### Responses

#### 200


Successfully seeked to the specified frame.


[FrameIndexResponse](#frameindexresponse)







#### 400


Invalid frame index, source does not support seeking, or source is not connected.




#### 404


Project or source not found.




#### 500


Unexpected error occurred.




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /api/v1/projects/{project_id}/offer

Create Webrtc Offer

Create a WebRTC offer


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| project_id | string | True |  |


### Request Body

[Offer](#offer)







### Responses

#### 200


WebRTC Answer


[Answer](#answer)







#### 400


Pipeline Not Active




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







# Components



## AnnotationSchema-Input



| Field | Type | Description |
|-------|------|-------------|
| config |  |  |
| label_id | string | Label for the annotation |


## AnnotationSchema-Output



| Field | Type | Description |
|-------|------|-------------|
| config |  |  |
| label_id | string | Label for the annotation |


## Answer



| Field | Type | Description |
|-------|------|-------------|
| sdp | string |  |
| type | string |  |


## FrameIndexResponse


Response for current frame index.


| Field | Type | Description |
|-------|------|-------------|
| index | integer |  |


## FrameListResponse


Paginated response for frame listing.


| Field | Type | Description |
|-------|------|-------------|
| frames | array |  |
| pagination |  |  |


## FrameMetadata


Metadata for a single frame in the timeline.


| Field | Type | Description |
|-------|------|-------------|
| index | integer |  |
| thumbnail | string |  |


## HTTPValidationError



| Field | Type | Description |
|-------|------|-------------|
| detail | array |  |


## ImagesFolderConfig



| Field | Type | Description |
|-------|------|-------------|
| source_type | string |  |
| images_folder_path | string |  |
| seekable | boolean |  |


## LabelCreateSchema



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| name | string | Label name |
| color |  | New hex color code, e.g. #RRGGBB |


## LabelSchema



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| name | string | Label name |
| color | string | New hex color code, e.g. #RRGGBB |


## LabelUpdateSchema



| Field | Type | Description |
|-------|------|-------------|
| name |  | Label name |
| color |  | New hex color code, e.g. #RRGGBB |


## LabelsListSchema



| Field | Type | Description |
|-------|------|-------------|
| labels | array |  |
| pagination |  |  |


## MatcherConfig



| Field | Type | Description |
|-------|------|-------------|
| model_type | string |  |
| num_foreground_points | integer |  |
| num_background_points | integer |  |
| mask_similarity_threshold | number |  |
| precision | string |  |


## Offer



| Field | Type | Description |
|-------|------|-------------|
| webrtc_id | string |  |
| sdp | string |  |
| type | string |  |


## Pagination


Pagination model.


| Field | Type | Description |
|-------|------|-------------|
| count | integer |  |
| total | integer |  |
| offset | integer |  |
| limit | integer |  |


## Point



| Field | Type | Description |
|-------|------|-------------|
| x | number | x coordinate |
| y | number | y coordinate |


## PolygonAnnotation



| Field | Type | Description |
|-------|------|-------------|
| type | string |  |
| points | array | Points defining the polygon |


## ProcessorCreateSchema



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| config |  |  |
| active | boolean |  |
| name | string |  |


## ProcessorListSchema



| Field | Type | Description |
|-------|------|-------------|
| models | array |  |
| pagination |  |  |


## ProcessorSchema



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| config |  |  |
| active | boolean |  |
| name | string |  |


## ProcessorUpdateSchema



| Field | Type | Description |
|-------|------|-------------|
| config |  |  |
| active | boolean |  |
| name | string |  |


## ProjectCreateSchema



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| name | string |  |


## ProjectSchema



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| name | string |  |
| active | boolean |  |


## ProjectUpdateSchema



| Field | Type | Description |
|-------|------|-------------|
| name |  |  |
| active |  |  |


## ProjectsListSchema



| Field | Type | Description |
|-------|------|-------------|
| projects | array |  |
| pagination |  |  |


## PromptsListSchema


Schema for listing prompts.


| Field | Type | Description |
|-------|------|-------------|
| prompts | array |  |
| pagination |  |  |


## RectangleAnnotation



| Field | Type | Description |
|-------|------|-------------|
| type | string |  |
| points | array | Two points defining rectangle: top-left and bottom-right |


## SourceCreateSchema



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| connected | boolean |  |
| config |  |  |


## SourceSchema



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| connected | boolean |  |
| config |  |  |


## SourceUpdateSchema



| Field | Type | Description |
|-------|------|-------------|
| connected | boolean |  |
| config |  |  |


## SourcesListSchema



| Field | Type | Description |
|-------|------|-------------|
| sources | array |  |


## TextPromptCreateSchema


Schema for creating a text prompt.


| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| type | string |  |
| content | string | Text content of the prompt |


## TextPromptSchema


Schema for a text prompt response.


| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| type | string |  |
| content | string |  |


## TextPromptUpdateSchema


Schema for updating a text prompt.


| Field | Type | Description |
|-------|------|-------------|
| type | string |  |
| content |  | Text content of the prompt |


## ValidationError



| Field | Type | Description |
|-------|------|-------------|
| loc | array |  |
| msg | string |  |
| type | string |  |


## VideoFileConfig



| Field | Type | Description |
|-------|------|-------------|
| source_type | string |  |
| video_path | string |  |
| seekable | boolean |  |


## VisualPromptCreateSchema


Schema for creating a visual prompt.


| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| type | string |  |
| frame_id | string | ID of the frame to use for the prompt |
| annotations | array | List of annotations for the prompt |


## VisualPromptListItemSchema


Schema for a visual prompt in list response (includes thumbnail).


| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| type | string |  |
| frame_id | string |  |
| annotations | array |  |
| thumbnail | string | Base64-encoded thumbnail image with annotations |


## VisualPromptSchema


Schema for a visual prompt response.


| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| type | string |  |
| frame_id | string |  |
| annotations | array |  |


## VisualPromptUpdateSchema


Schema for updating a visual prompt.


| Field | Type | Description |
|-------|------|-------------|
| type | string |  |
| frame_id |  | ID of the frame to use for the prompt |
| annotations |  | List of annotations for the prompt |


## WebCamConfig



| Field | Type | Description |
|-------|------|-------------|
| source_type | string |  |
| device_id | integer |  |
| seekable | boolean |  |
