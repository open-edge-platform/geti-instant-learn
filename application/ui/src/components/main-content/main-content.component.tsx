/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { NoMedia } from '@geti-prompt/icons';
import { Content, Flex, View } from '@geti/ui';

import { useCurrentProject } from '../../features/project/hooks/use-current-project.hook';
import { NotActiveProject } from '../../features/project/not-active-project/not-active-project.component';
import { useGetSources } from '../../features/sources-sinks-configuration/sources-configuration/hooks/use-get-sources';
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
    const { data } = useCurrentProject();
    const { data: sourcesData } = useGetSources();

    if (!data.active) {
        return <NotActiveProject project={data} />;
    }

    if (sourcesData.sources.length === 0) {
        return <NoSourcePlaceholder />;
    }

    return <StreamContainer />;
};
