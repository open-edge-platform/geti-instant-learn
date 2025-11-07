/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { components, type SchemaSourceSchema, type SchemaSourcesListSchema } from './openapi-spec';

export type SourcesListType = SchemaSourcesListSchema['sources'];
export type SourceType = SourcesListType[number]['config']['source_type'];

type SourceWithoutConfig = Omit<SchemaSourceSchema, 'config'>;

type WebcamConfig = components['schemas']['WebCamConfig'];
type VideoFileConfig = components['schemas']['VideoFileConfig'];
export type ImagesFolderConfig = components['schemas']['ImagesFolderConfig'];

export type SourceConfig = WebcamConfig | VideoFileConfig | ImagesFolderConfig;

export type WebcamSourceType = SourceWithoutConfig & { config: WebcamConfig };
export type VideoFileSourceType = SourceWithoutConfig & { config: VideoFileConfig };
export type ImagesFolderSourceType = SourceWithoutConfig & { config: ImagesFolderConfig };

export { $api, client } from './client';
export {
    type SchemaProjectSchema as ProjectType,
    type SchemaProjectsListSchema as ProjectsListType,
    type SchemaProjectUpdateSchema as ProjectUpdateType,
    type SchemaSourceSchema as Source,
    type SchemaLabelSchema as LabelType,
    type SchemaLabelsListSchema as LabelListType,
    type paths,
    type SchemaSourceCreateSchema as SourceCreateType,
    type SchemaSourceUpdateSchema as SourceUpdateType,
} from './openapi-spec';
