/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect } from 'react';

import { useCurrentProject, useProjectIdentifier } from '@geti-prompt/hooks';
import { Grid, minmax, View } from '@geti/ui';

import { Header } from '../components/header/header.component';
import { MainContent } from '../components/main-content/main-content.component';
import { Sidebar } from '../components/sidebar/sidebar.component';
import { Toolbar } from '../components/toolbar/toolbar.component';
import { paths } from '../constants/paths';
import { FullScreenModeProvider } from '../features/annotator/actions/full-screen-mode.component';
import { useActivateProject } from '../features/project/api/use-activate-project.hook';
import { ProjectsListPanel } from '../features/project/projects-list-panel.component';
import { WebRTCConnectionProvider } from '../features/stream/web-rtc/web-rtc-connection-provider';
import { SelectedFrameProvider } from '../shared/selected-frame-provider.component';

const useEnsureValidAndActiveProject = () => {
    // Check if the current project is valid, if it's not error boundary will catch it.
    const { data } = useCurrentProject();

    const activateProject = useActivateProject();

    useEffect(() => {
        if (!data.active) {
            activateProject.mutate(data);
        }
        // We only want to activate the project when a project that is being open is not active.
        // This might happen only when a user opens a link to a project that is not active.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [data.active]);
};

export const ProjectRoute = () => {
    const { projectId } = useProjectIdentifier();

    useEnsureValidAndActiveProject();

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
                    <FullScreenModeProvider>
                        <View backgroundColor={'gray-50'} gridArea={'main'}>
                            <MainContent />
                        </View>

                        <Sidebar />
                    </FullScreenModeProvider>
                </SelectedFrameProvider>
            </Grid>
        </WebRTCConnectionProvider>
    );
};
