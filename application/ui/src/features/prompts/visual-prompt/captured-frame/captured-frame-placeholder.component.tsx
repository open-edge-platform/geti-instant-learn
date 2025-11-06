/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Image } from '@geti-prompt/icons';
import { Content, Flex } from '@geti/ui';

import styles from './captured-frame-placeholder.module.scss';

export const CapturedFramePlaceholder = () => {
    return (
        <Flex
            height={'100%'}
            width={'100%'}
            justifyContent={'center'}
            alignItems={'center'}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-300)',
            }}
        >
            <Flex direction={'column'} gap={'size-100'} alignItems={'center'} height={'100%'} justifyContent={'center'}>
                <Image />
                <Content UNSAFE_className={styles.noFramePlaceholder}>Capture frames for visual prompt</Content>
            </Flex>
        </Flex>
    );
};
