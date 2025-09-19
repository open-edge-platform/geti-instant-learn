/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { Navigate } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';
import { path } from 'static-path';

import { MainContent } from '../components/main-content.component';
import { ErrorPage } from './error-page.component';
import { ProjectLayout } from './project-layout.component';

export const routes = {
    root: path('/'),
    project: path('/projects/:projectId'),
};

const RedirectToProject = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');

    return <Navigate to={routes.project({ projectId: data.projects[0].id })} replace />;
};

export const router = createBrowserRouter([
    {
        path: routes.root.pattern,
        element: <RedirectToProject />,
    },
    {
        path: routes.project.pattern,
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
