/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect } from 'react';

import { useCurrentProject, useProjectIdentifier } from '@geti-prompt/hooks';
import { Flex, Grid, minmax, View } from '@geti/ui';
import { Group, Panel } from 'react-resizable-panels';

import { Header } from '../components/header/header.component';
import { MainContent } from '../components/main-content/main-content.component';
import { Sidebar } from '../components/sidebar/sidebar.component';
import { Toolbar } from '../components/toolbar/toolbar.component';
import { paths } from '../constants/paths';
import { useActivateProject } from '../features/project/api/use-activate-project.hook';
import { ProjectsListPanel } from '../features/project/projects-list-panel.component';
import { WebRTCConnectionProvider } from '../features/stream/web-rtc/web-rtc-connection-provider';
import { SelectedFrameProvider } from '../shared/selected-frame-provider.component';

const MainLayout = () => {
    return (
        <Group>
            <Panel minSize={'30%'} id={'main'}>
                <Flex direction={'column'} height={'100%'}>
                    <Toolbar />

                    <View backgroundColor={'gray-50'} flex={1} minHeight={0}>
                        <MainContent />
                    </View>
                </Flex>
            </Panel>

            <Sidebar />
        </Group>
    );
};

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
    useEnsureValidAndActiveProject();

    const { projectId } = useProjectIdentifier();

    return (
        <WebRTCConnectionProvider key={projectId}>
            <Grid areas={['header', 'main']} rows={['size-800', minmax(0, '1fr')]} columns={'1fr'} height={'100vh'}>
                <Header homeLink={paths.projects({})}>
                    <ProjectsListPanel />
                </Header>

                <SelectedFrameProvider>
                    <MainLayout />
                </SelectedFrameProvider>
            </Grid>
        </WebRTCConnectionProvider>
    );
};
