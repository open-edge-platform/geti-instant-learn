/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Grid, minmax, View } from '@geti/ui';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';

import { Header } from '../components/header/header.component';
import { MainContent } from '../components/main-content.component';
import { Sidebar } from '../components/sidebar/sidebar.component';
import { Toolbar } from '../components/toolbar.component';
import { useCurrentProject } from '../features/project/hooks/use-current-project.hook';
import { SelectedFrameProvider } from '../features/stream/selected-frame-provider.component';
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
                areas={['header', 'toolbar', 'main']}
                rows={['size-800', 'size-700', minmax(0, '1fr')]}
                columns={['1fr']}
                height={'100vh'}
            >
                <Header />

                <Toolbar />

                <SelectedFrameProvider>
                    <View gridArea={'main'}>
                        <PanelGroup direction='horizontal' style={{ height: '100%', width: '100%' }}>
                            <Panel defaultSize={50} minSize={30} style={{ overflow: 'auto' }}>
                                <View backgroundColor={'gray-50'} height={'100%'} width={'100%'}>
                                    <MainContent />
                                </View>
                            </Panel>
                            <PanelResizeHandle
                                style={{
                                    width: 'var(--spectrum-global-dimension-size-50)',
                                    background: 'var(--spectrum-global-color-gray-400)',
                                }}
                            />
                            <Panel defaultSize={50} minSize={30} style={{ overflow: 'auto' }}>
                                <Sidebar />
                            </Panel>
                        </PanelGroup>
                    </View>
                </SelectedFrameProvider>
            </Grid>
        </WebRTCConnectionProvider>
    );
};
