/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, Suspense } from 'react';

import { Flex, Item, Loading, Picker, Text, View } from '@geti/ui';

import { useGetModels } from '../api/use-get-models';
import { useUpdateModel } from '../api/use-update-model';
import { ModelConfiguration } from './model-configuration/model-configuration.component';

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
    const models = useGetModels();
    const updateModel = useUpdateModel();
    const activeModel = models.find((model) => model.active) ?? models[0];

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
                defaultSelectedKey={activeModel.id}
                onSelectionChange={handleSelectionChange}
                items={models}
            >
                {(item) => <Item key={item.id}>{item.name}</Item>}
            </Picker>

            <ModelConfiguration model={activeModel} />
        </Flex>
    );
};
