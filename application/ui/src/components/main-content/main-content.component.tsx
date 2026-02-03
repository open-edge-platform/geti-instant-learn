/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useGetSources } from '@/hooks';
import { NoMedia } from '@/icons';
import { Content, Flex, View } from '@geti/ui';

import { StreamContainer } from '../../features/stream/stream-container/stream-container.component';

import styles from './main-content.module.scss';

const NoSourcePlaceholder = () => {
    return (
        <View paddingX={'size-800'} paddingY={'size-1000'} height={'100%'}>
            <View backgroundColor={'gray-200'} height={'100%'} UNSAFE_className={styles.container}>
                <Flex height={'100%'} width={'100%'} justifyContent={'center'} alignItems={'center'}>
                    <Flex direction={'column'} gap={'size-100'} alignItems={'center'}>
                        <View>
                            <NoMedia />
                        </View>
                        <Content UNSAFE_className={styles.title}>Setup your input source</Content>
                    </Flex>
                </Flex>
            </View>
        </View>
    );
};

export const MainContent = () => {
    const { data: sourcesData } = useGetSources();

    if (sourcesData.sources.length === 0) {
        return <NoSourcePlaceholder />;
    }

    return <StreamContainer />;
};
