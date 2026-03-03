/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelType } from '@/api';

import { useGetModels } from '../../prompts/models/api/use-get-models';
import { getAnnotationTypeForModel } from '../../prompts/models/utils';
import { RectangleTool } from './rectangle-tool/rectangle-tool.component';
import { SegmentAnythingTool } from './segment-anything-tool/segment-anything-tool.component';

export const ToolManager = () => {
    const models = useGetModels();
    const activeModel = models.find((m: ModelType) => m.active) ?? models[0];
    const annotationType = getAnnotationTypeForModel(activeModel);

    return annotationType === 'rectangle' ? <RectangleTool /> : <SegmentAnythingTool />;
};
