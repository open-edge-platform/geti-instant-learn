/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { Navigate } from 'react-router';
import { createBrowserRouter } from 'react-router-dom';

import { MainContent } from '../components/main-content.component';
import { ProjectsListEntry } from '../features/projects-management/projects-list-entry/projects-list-entry.component';
import { Welcome } from '../features/projects-management/projects-list-entry/welcome.component';
import { ErrorPage } from './error-page.component';
import { paths } from './paths';
import { ProjectLayout } from './project-layout.component';
import { RootLayout } from './root-layout.component';

const Redirect = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');

    const projects = data.projects;

    if (projects.length === 0) {
        return <Navigate to={paths.welcome({})} replace />;
    }

    if (projects.length === 1) {
        return <Navigate to={paths.project({ projectId: data.projects[0].id })} replace />;
    }

    const activeProject = projects.find((project) => project.active);

    if (activeProject === undefined) {
        return <Navigate to={paths.projects({})} replace />;
    }

    return <Navigate to={paths.project({ projectId: activeProject.id })} replace />;
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        element: <RootLayout />,
        errorElement: <ErrorPage />,
        children: [
            {
                index: true,
                element: <Redirect />,
            },
            {
                path: paths.welcome.pattern,
                element: <Welcome />,
            },
            {
                path: paths.projects.pattern,
                element: <ProjectsListEntry />,
            },
            {
                path: paths.project.pattern,
                element: <ProjectLayout />,
                children: [
                    {
                        index: true,
                        element: <MainContent />,
                    },
                ],
            },
            {
                path: '*',
                element: <Redirect />,
            },
        ],
    },
]);
