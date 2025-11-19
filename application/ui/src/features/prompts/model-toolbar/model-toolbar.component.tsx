/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, useState } from 'react';

import { Flex, Item, Picker, Text } from '@geti/ui';

import styles from './model-toolbar.module.scss';

const useModels = () => {
    // TODO: replace with actual data
    const mockDate = new Date().toLocaleString('en-US', {
        day: '2-digit',
        month: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
    });

    return [
        { id: '1', name: 'DINO v2', deployedAt: mockDate },
        { id: '2', name: 'DINO v3', deployedAt: mockDate },
        { id: '3', name: 'DINO v4', deployedAt: mockDate },
    ];
};

export const ModelToolbar = () => {
    const models = useModels();
    const [selectedModel, setSelectedModel] = useState(models[0]);

    const handleSelectionChange = (key: Key | null) => {
        const selected = models.find((model) => model.id === key);

        if (selected) {
            setSelectedModel(selected);
        }
    };

    return (
        <Flex alignItems={'end'} gap={'size-100'} justifyContent={'space-between'}>
            <Picker
                label={'Model'}
                defaultSelectedKey={selectedModel.id}
                onSelectionChange={handleSelectionChange}
                items={models}
            >
                {(item) => <Item key={item.id}>{item.name}</Item>}
            </Picker>

            <Flex UNSAFE_className={styles.deployedText}>
                <Text>Deployed: {selectedModel.deployedAt}</Text>
            </Flex>
        </Flex>
    );
};
