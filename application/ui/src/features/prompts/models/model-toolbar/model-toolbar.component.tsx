/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key } from 'react';

import { Flex, Item, Picker, Text } from '@geti/ui';

import { useGetModels } from '../api/use-get-models';
import { useSetActiveModel } from '../api/use-set-active-model';

import styles from './model-toolbar.module.scss';

export const ModelToolbar = () => {
    const models = useGetModels();
    const setActiveModel = useSetActiveModel();
    const activeModel = models.find((model) => model.active) || models[0];

    const handleSelectionChange = (key: Key | null) => {
        const selectedModel = models.find((model) => model.id === key);

        if (selectedModel) {
            setActiveModel(selectedModel);
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
        <Flex alignItems={'end'} gap={'size-100'} justifyContent={'space-between'}>
            <Picker
                label={'Model'}
                defaultSelectedKey={activeModel.id}
                onSelectionChange={handleSelectionChange}
                items={models}
            >
                {(item) => <Item key={item.id}>{item.name}</Item>}
            </Picker>

            <Flex UNSAFE_className={styles.deployedText}>
                <Text>Deployed: at some point</Text>
                {/* TODO: Enable for development only. We dont have a design for model creation */}
                {/* <Button onPress={() => createModel()}>Create Model</Button> */}
            </Flex>
        </Flex>
    );
};
