/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { View } from '@geti/ui';

import { useCurrentProject } from '../features/project/hooks/use-current-project.hook';
import { NotActiveProject } from '../features/project/not-active-project/not-active-project.component';
import { useGetSources } from '../features/sources-sinks-configuration/sources-configuration/hooks/use-get-sources';
import { StreamContainer } from '../features/stream/stream-container/stream-container.component';
import { NoMediaPlaceholder } from './no-media-placeholder/no-media-placeholder.component';

const NoSourcePlaceholder = () => {
    return (
        <View paddingX={'size-800'} paddingY={'size-1000'} height={'100%'}>
            <NoMediaPlaceholder title={'Setup your input source'} />
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
