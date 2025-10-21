/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { View } from '@geti/ui';

import { useCurrentProject } from '../features/projects-management/hooks/use-current-project.hook';
import { NotActiveProject } from '../features/stream/not-active-project/not-active-project.component';
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

    if (!data.active) {
        return <NotActiveProject project={data} />;
    }

    return <NoSourcePlaceholder />;
};
