/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, Item, Picker, Text } from '@geti/ui';

import styles from './model-toolbar.module.scss';

const useModels = () => {
    // TODO: replace with actual data

    const mockDate = new Date().toLocaleString('en-US', {
        day: '2-digit',
        month: 'long',
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

    return (
        <Flex alignItems={'end'} gap={'size-100'}>
            <Picker label={'Model'} defaultSelectedKey={models[0].id} items={models}>
                {(item) => <Item key={item.id}>{item.name}</Item>}
            </Picker>

            <Flex UNSAFE_className={styles.deployedText}>
                <Text>Deployed: {models[0].deployedAt}</Text>
            </Flex>
        </Flex>
    );
};
