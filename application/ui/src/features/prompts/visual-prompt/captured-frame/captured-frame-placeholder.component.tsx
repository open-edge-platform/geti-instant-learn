/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Image } from '@geti-prompt/icons';
import { Content, Flex, View } from '@geti/ui';

import styles from './captured-frame-placeholder.module.scss';

export const CapturedFramePlaceholder = () => {
    return (
        <View backgroundColor={'gray-300'} height={'size-6000'}>
            <Flex height={'100%'} width={'100%'} justifyContent={'center'} alignItems={'center'}>
                <Flex direction={'column'} gap={'size-100'} alignItems={'center'}>
                    <Image />
                    <Content UNSAFE_className={styles.noFramePlaceholder}>Capture frames for visual prompt</Content>
                </Flex>
            </Flex>
        </View>
    );
};
