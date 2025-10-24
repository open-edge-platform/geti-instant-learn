/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, Text } from '@geti/ui';

import styles from './model-toolbar.module.scss';

export const ModelToolbar = () => {
    return (
        <Flex alignItems={'baseline'} justifyContent={'space-between'}>
            <Text>DINO v2</Text>
            <Text UNSAFE_className={styles.deployedText}>
                Deployed:{' '}
                {new Date().toLocaleString('en-US', {
                    day: '2-digit',
                    month: 'long',
                    year: 'numeric',
                    hour: 'numeric',
                    minute: '2-digit',
                    hour12: true,
                })}
            </Text>
        </Flex>
    );
};
