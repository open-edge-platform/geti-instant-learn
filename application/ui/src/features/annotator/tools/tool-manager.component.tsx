/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelType } from '@/api';

import { useGetModels } from '../../prompts/models/api/use-get-models';
import { useSupportedPromptTypesMap } from '../../prompts/models/use-supported-prompt-types';
import { getAnnotationTypeFromPromptTypes } from '../../prompts/models/utils';
import { BoundingBoxTool } from './boundingbox-tool/boundigbox-tool.component';
import { SegmentAnythingTool } from './segment-anything-tool/segment-anything-tool.component';

export const ToolManager = () => {
    const models = useGetModels();
    const promptTypesMap = useSupportedPromptTypesMap();
    const activeModel = models.find((m: ModelType) => m.active) ?? models[0];

    if (!activeModel) {
        return null;
    }

    const supportedTypes = promptTypesMap.get(activeModel.config.model_type) ?? [];
    const annotationType = getAnnotationTypeFromPromptTypes(supportedTypes);

    return annotationType === 'rectangle' ? <BoundingBoxTool /> : <SegmentAnythingTool />;
};
