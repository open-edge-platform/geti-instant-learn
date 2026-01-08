/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCurrentProject, useProjectIdentifier } from '@geti-prompt/hooks';
import { dimensionValue, Flex, Grid, minmax, View } from '@geti/ui';
import { Group, Panel, Separator, useDefaultLayout } from 'react-resizable-panels';

import { Header } from '../../components/header/header.component';
import { MainContent } from '../../components/main-content/main-content.component';
import { Sidebar } from '../../components/sidebar/sidebar.component';
import { Toolbar } from '../../components/toolbar/toolbar.component';
import { paths } from '../../constants/paths';
import { ProjectsListPanel } from '../../features/project/projects-list-panel.component';
import { WebRTCConnectionProvider } from '../../features/stream/web-rtc/web-rtc-connection-provider';
import { SelectedFrameProvider } from '../../shared/selected-frame-provider.component';

import styles from './project-route.module.scss';

const MainLayout = () => {
    const { defaultLayout, onLayoutChange } = useDefaultLayout({
        id: 'stream-sidebar-layout',
        storage: localStorage,
    });

    return (
        <Group defaultLayout={defaultLayout} onLayoutChange={onLayoutChange}>
            <Panel minSize={'30%'}>
                <Flex direction={'column'} height={'100%'}>
                    <Toolbar />

                    <View backgroundColor={'gray-50'} flex={1} minHeight={0}>
                        <MainContent />
                    </View>
                </Flex>
            </Panel>
            <Separator className={styles.separator} />
            {/* 48px is a size of the tiny sidebar that toggles sidebar content */}
            <Panel minSize={'48px'}>
                <Sidebar />
            </Panel>
        </Group>
    );
};

export const ProjectRoute = () => {
    // Check if the current project is valid
    useCurrentProject();

    const { projectId } = useProjectIdentifier();

    return (
        <WebRTCConnectionProvider key={projectId}>
            <Grid
                areas={['header', 'main']}
                rows={['size-800', minmax(0, '1fr')]}
                columns={[minmax('50%', '1fr')]}
                height={'100vh'}
            >
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
