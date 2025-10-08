/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Grid, minmax } from '@geti/ui';
import { Outlet } from 'react-router';

import { Header } from '../components/header.component';
import { Sidebar } from '../components/sidebar/sidebar.component';
import { Toolbar } from '../components/toolbar.component';

const useCheckIfProjectIsValid = () => {
    const { projectId } = useProjectIdentifier();

    // Note: when the project is not found, the project query will throw an error and the parent error boundary with
    // ErrorPage will be rendered
    $api.useSuspenseQuery(
        'get',
        '/api/v1/projects/{project_id}',
        {
            params: {
                path: {
                    project_id: projectId,
                },
            },
        },
        {
            retry: 1,
        }
    );
};

export const ProjectLayout = () => {
    useCheckIfProjectIsValid();

    return (
        <Grid
            areas={['header header header', 'toolbar prompt-sidebar sidebar', 'main prompt-sidebar sidebar']}
            rows={['size-800', 'size-700', '1fr']}
            columns={[minmax('50%', '1fr'), 'auto']}
            height={'100vh'}
        >
            <Header />

            <Toolbar />

            <Outlet />

            <Sidebar />
        </Grid>
    );
};
