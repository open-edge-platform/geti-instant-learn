/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, Suspense, useEffect } from 'react';

import { ModelType } from '@/api';
import { usePromptMode, type PromptMode } from '@/hooks';
import { Flex, Item, Loading, Picker, Text, View } from '@geti/ui';

import { useGetModels } from '../api/use-get-models';
import { type SupportedPromptType } from '../api/use-get-supported-models';
import { useUpdateModel } from '../api/use-update-model';
import { useSupportedPromptTypesMap } from '../use-supported-prompt-types';
import { ModelConfiguration } from './model-configuration/model-configuration.component';

const isModelCompatible = (
    model: ModelType,
    promptMode: PromptMode,
    promptTypesMap: Map<string, SupportedPromptType[]>
): boolean => {
    const supportedTypes = promptTypesMap.get(model.config.model_type) ?? [];
    if (promptMode === 'visual') {
        return supportedTypes.some((t) => t.startsWith('visual'));
    }
    return supportedTypes.includes('text');
};

export const ModelToolbar = () => {
    return (
        <View position={'relative'} minHeight={'size-700'}>
            <Suspense fallback={<Loading size={'M'} />}>
                <ModelToolbarContent />
            </Suspense>
        </View>
    );
};

const ModelToolbarContent = () => {
    const allModels = useGetModels();
    const updateModel = useUpdateModel();
    const [promptMode] = usePromptMode();
    const promptTypesMap = useSupportedPromptTypesMap();

    // Filter models compatible with the current prompt mode
    const models = allModels.filter((model) => isModelCompatible(model, promptMode, promptTypesMap));

    const activeModel = models.find((m) => m.active) ?? models[0];

    // Auto-select the first compatible model when the active model is
    // incompatible with the current prompt mode. Covers both prompt-mode
    // switches and initial mount where the backend may have stored an
    // incompatible model as active.
    useEffect(() => {
        if (models.length === 0) {
            return;
        }

        const currentActiveModel = allModels.find((m) => m.active);

        if (currentActiveModel && isModelCompatible(currentActiveModel, promptMode, promptTypesMap)) {
            return;
        }

        if (updateModel.isPending) {
            return;
        }

        updateModel.mutate({ ...models[0], active: true });
    }, [promptMode, models, allModels, promptTypesMap, updateModel]);

    const handleSelectionChange = (key: Key | null) => {
        const selectedModel = models.find((model) => model.id === key);

        if (selectedModel) {
            updateModel.mutate({ ...selectedModel, active: true });
        }
    };

    if (models.length === 0) {
        return (
            <Flex alignItems={'center'}>
                <Text>No models available</Text>
            </Flex>
        );
    }

    return (
        <Flex alignItems={'end'} gap={'size-100'}>
            <Picker
                isQuiet
                label={'Model'}
                selectedKey={activeModel?.id}
                onSelectionChange={handleSelectionChange}
                items={models}
            >
                {(item) => <Item key={item.id}>{item.name}</Item>}
            </Picker>

            {activeModel && <ModelConfiguration model={activeModel} />}
        </Flex>
    );
};
