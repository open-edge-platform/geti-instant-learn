/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Grid, minmax, View } from '@geti/ui';

import { Header } from '../components/header/header.component';
import { MainContent } from '../components/main-content.component';
import { Sidebar } from '../components/sidebar/sidebar.component';
import { Toolbar } from '../components/toolbar.component';
import { useCurrentProject } from '../features/projects-management/hooks/use-current-project.hook';
import { WebRTCConnectionProvider } from '../features/stream/web-rtc/web-rtc-connection-provider';

const useCheckIfProjectIsValid = () => {
    useCurrentProject();
};

export const ProjectLayout = () => {
    useCheckIfProjectIsValid();

    const { projectId } = useProjectIdentifier();

    return (
        <WebRTCConnectionProvider key={projectId}>
            <Grid
                areas={['header header header', 'toolbar prompt-sidebar sidebar', 'main prompt-sidebar sidebar']}
                rows={['size-800', 'size-700', minmax(0, '1fr')]}
                columns={[minmax('50%', '1fr'), 'auto']}
                height={'100vh'}
            >
                <Header />

                <Toolbar />

                <View backgroundColor={'gray-50'} gridArea={'main'}>
                    <MainContent />
                </View>

                <Sidebar />
            </Grid>
        </WebRTCConnectionProvider>
    );
};
