/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { Navigate } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';

import { MainContent } from '../components/main-content.component';
import { ErrorPage } from './error-page.component';
import { paths } from './paths';
import { ProjectLayout } from './project-layout.component';

const RedirectToProject = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');

    return <Navigate to={paths.project({ projectId: data.projects[0].id })} replace />;
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        element: <RedirectToProject />,
    },
    {
        path: paths.project.pattern,
        errorElement: <ErrorPage />,
        element: <ProjectLayout />,
        children: [
            {
                index: true,
                element: <MainContent />,
            },
        ],
    },
]);
