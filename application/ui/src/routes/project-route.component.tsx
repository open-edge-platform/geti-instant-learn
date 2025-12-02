/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCurrentProject, useProjectIdentifier } from '@geti-prompt/hooks';
import { Grid, minmax, View } from '@geti/ui';

import { Header } from '../components/header/header.component';
import { MainContent } from '../components/main-content/main-content.component';
import { Sidebar } from '../components/sidebar/sidebar.component';
import { Toolbar } from '../components/toolbar/toolbar.component';
import { paths } from '../constants/paths';
import { ProjectsListPanel } from '../features/project/projects-list-panel.component';
import { WebRTCConnectionProvider } from '../features/stream/web-rtc/web-rtc-connection-provider';
import { SelectedFrameProvider } from '../shared/selected-frame-provider.component';

export const ProjectRoute = () => {
    // Check if the current project is valid
    useCurrentProject();

    const { projectId } = useProjectIdentifier();

    return (
        <WebRTCConnectionProvider key={projectId}>
            <Grid
                areas={['header header header', 'toolbar prompt-sidebar sidebar', 'main prompt-sidebar sidebar']}
                rows={['size-800', 'size-700', minmax(0, '1fr')]}
                columns={[minmax('50%', '1fr'), 'auto']}
                height={'100vh'}
            >
                <Header homeLink={paths.projects({})}>
                    <ProjectsListPanel />
                </Header>

                <Toolbar />

                <SelectedFrameProvider>
                    <View backgroundColor={'gray-50'} gridArea={'main'}>
                        <MainContent />
                    </View>

                    <Sidebar />
                </SelectedFrameProvider>
            </Grid>
        </WebRTCConnectionProvider>
    );
};
