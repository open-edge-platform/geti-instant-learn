/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useGetSupportedModels, type SupportedPromptType } from './api/use-get-supported-models';

/**
 * Build a map from model_type → supported_prompt_types using the
 * /api/v1/system/supported-models endpoint response.
 */
export const useSupportedPromptTypesMap = (): Map<string, SupportedPromptType[]> => {
    const supportedModels = useGetSupportedModels();

    return new Map(supportedModels.map((m) => [m.default_config.model_type, m.supported_prompt_types]));
};
