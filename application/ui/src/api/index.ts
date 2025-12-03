/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { components, type SchemaSourceSchema, type SchemaSourcesListSchema } from './openapi-spec';

export type SourcesType = SchemaSourcesListSchema['sources'];
export type SourceType = SourcesType[number]['config']['source_type'];

type SourceWithoutConfig = Omit<SchemaSourceSchema, 'config'>;

type WebcamConfig = components['schemas']['WebCamConfig'];
type VideoFileConfig = components['schemas']['VideoFileConfig'];
export type ImagesFolderConfig = components['schemas']['ImagesFolderConfig'];
type SampleDatasetConfig = components['schemas']['SampleDatasetConfig'];

export type SourceConfig = WebcamConfig | VideoFileConfig | ImagesFolderConfig | SampleDatasetConfig;

export type WebcamSourceType = SourceWithoutConfig & { config: WebcamConfig };
export type VideoFileSourceType = SourceWithoutConfig & { config: VideoFileConfig };
export type ImagesFolderSourceType = SourceWithoutConfig & { config: ImagesFolderConfig };
export type SampleDatasetSourceType = SourceWithoutConfig & { config: SampleDatasetConfig };

export type SinkType = components['schemas']['SinkSchema'];
type SinkWithoutConfig = Omit<SinkType, 'config'>;

export type MQTTConfig = components['schemas']['MqttConfig'];
export type MQTTSinkType = SinkWithoutConfig & { config: MQTTConfig };

export { $api, client } from './client';
export {
    type paths,
    type SchemaProcessorSchema as ModelType,
    type SchemaProcessorListSchema as ModelListType,
    type SchemaProjectSchema as ProjectType,
    type SchemaProjectsListSchema as ProjectsListType,
    type SchemaProjectUpdateSchema as ProjectUpdateType,
    type SchemaSourceSchema as Source,
    type SchemaLabelSchema as LabelType,
    type SchemaLabelsListSchema as LabelListType,
    type SchemaSourcesListSchema as SourcesListType,
    type SchemaSourceCreateSchema as SourceCreateType,
    type SchemaSourceUpdateSchema as SourceUpdateType,
    type SchemaAnnotationSchemaInput as AnnotationPostType,
    type SchemaAnnotationSchemaOutput as AnnotationType,
    type SchemaVisualPromptSchema as VisualPromptType,
    type SchemaPromptsListSchema as VisualPromptListType,
    type SchemaVisualPromptListItemSchema as VisualPromptItemType,
    type SchemaFrameMetadata as FrameAPIType,
    type SchemaFrameListResponse as FramesResponseType,
    type SchemaSinkCreateSchema as SinkCreateType,
    type SchemaSinkUpdateSchema as SinkUpdateType,
    type SchemaSinksListSchema as SinksListType,
} from './openapi-spec';
