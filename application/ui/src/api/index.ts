/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { components, type SchemaSourceSchema, type SchemaSourcesListSchema } from './openapi-spec';

export type SourcesListType = SchemaSourcesListSchema['sources'];
export type SourceType = SourcesListType[number]['config']['source_type'];

type SourceWithoutConfig = Omit<SchemaSourceSchema, 'config'>;

export type WebcamConfig = SourceWithoutConfig & { config: components['schemas']['WebCamConfig'] };
export type VideoFileConfig = SourceWithoutConfig & { config: components['schemas']['VideoFileConfig'] };
export type ImagesFolderConfig = SourceWithoutConfig & { config: components['schemas']['ImagesFolderConfig'] };

export { $api, client } from './client';
export {
    type SchemaProjectSchema as ProjectType,
    type SchemaProjectsListSchema as ProjectsListType,
    type SchemaProjectUpdateSchema as ProjectUpdateType,
    type SchemaSourceSchema as SourceConfig,
    type SchemaLabelSchema as LabelType,
    type SchemaLabelsListSchema as LabelListType,
    type paths,
} from './openapi-spec';
