/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { View } from '@geti/ui';

import { NoActiveProject } from '../features/stream/no-active-project/no-active-project.component';
import { NoMediaPlaceholder } from './no-media-placeholder/no-media-placeholder.component';

const NoSourcePlaceholder = () => {
    return (
        <View paddingX={'size-800'} paddingY={'size-1000'} height={'100%'}>
            <NoMediaPlaceholder title={'Setup your input source'} />
        </View>
    );
};

export const MainContent = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    if (!data.active) {
        return <NoActiveProject project={data} />;
    }

    return <NoSourcePlaceholder />;
};
