/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { View } from '@geti/ui';

import { useCurrentProject } from '../features/projects-management/hooks/use-current-project.hook';
import { NotActiveProject } from '../features/projects-management/not-active-project/not-active-project.component';
import { StreamContainer } from '../features/stream/stream.component';
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
    const { data: sourcesData } = $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}/sources', {
        params: {
            path: {
                project_id: data.id,
            },
        },
    });

    if (!data.active) {
        return <NotActiveProject project={data} />;
    }

    if (sourcesData.sources.length === 0) {
        return <NoSourcePlaceholder />;
    }

    return <StreamContainer />;
};
