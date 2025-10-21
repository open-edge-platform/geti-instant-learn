/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Grid, minmax, View } from '@geti/ui';
import { Outlet } from 'react-router';

import { Header } from '../components/header.component';
import { Sidebar } from '../components/sidebar/sidebar.component';
import { Toolbar } from '../components/toolbar.component';
import { useCurrentProject } from '../features/projects-management/hooks/use-current-project.hook';

const useCheckIfProjectIsValid = () => {
    useCurrentProject({
        retry: 1,
    });
};

export const ProjectLayout = () => {
    useCheckIfProjectIsValid();

    return (
        <Grid
            areas={['header header header', 'toolbar prompt-sidebar sidebar', 'main prompt-sidebar sidebar']}
            rows={['size-800', 'size-700', minmax(0, '1fr')]}
            columns={[minmax('50%', '1fr'), 'auto']}
            height={'100vh'}
        >
            <Header />

            <Toolbar />

            <View backgroundColor={'gray-50'} gridArea={'main'}>
                <Outlet />
            </View>

            <Sidebar />
        </Grid>
    );
};
