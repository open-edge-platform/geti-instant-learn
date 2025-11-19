/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { getMockedProject, render } from '@geti-prompt/test-utils';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../../setup-test';
import { NotActiveProject } from './not-active-project.component';

describe('NotActiveProject', () => {
    const inactiveProject = getMockedProject({
        id: 'inactive-project-id',
        name: 'I am very obsolete',
        active: false,
    });

    const activeProject = getMockedProject({
        id: 'active-project-id',
        name: 'Activation impossible',
        active: true,
    });

    beforeEach(() => {
        server.use(
            http.get('/api/v1/projects', () => {
                return HttpResponse.json({
                    projects: [inactiveProject, activeProject],
                    pagination: { total: 2, count: 2, offset: 0, limit: 10 },
                });
            })
        );
    });

    it('renders inactive project UIs', async () => {
        render(<NotActiveProject project={inactiveProject} />);

        expect(await screen.findByText(/This project is set as inactive/i)).toBeInTheDocument();
        expect(screen.getByText(/Would you like to activate this project/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Activate current project/i })).toBeInTheDocument();
    });

    it('opens confirmation dialog when activate is clicked and another project is active', async () => {
        render(<NotActiveProject project={inactiveProject} />);

        const activateButton = await screen.findByRole('button', { name: /Activate current project/i });
        fireEvent.click(activateButton);

        expect(await screen.findByRole('heading', { name: /Activate project/i })).toBeInTheDocument();
        expect(screen.getByText('I am very obsolete')).toBeInTheDocument();
        expect(screen.getByText('Activation impossible')).toBeInTheDocument();

        const cancelButton = screen.getByRole('button', { name: /Cancel/i });
        fireEvent.click(cancelButton);

        await waitFor(() => {
            expect(screen.queryByRole('heading', { name: /Activate project/i })).not.toBeInTheDocument();
        });
    });

    it('activates project when confirmation dialog activate button is clicked', async () => {
        let updateRequestMade = false;
        server.use(
            http.put('/api/v1/projects/{project_id}', async ({ request }) => {
                const body = await request.json();
                updateRequestMade = true;

                expect(body).toEqual({ active: true });

                return HttpResponse.json({
                    ...inactiveProject,
                    active: true,
                });
            })
        );

        render(<NotActiveProject project={inactiveProject} />);

        const activateButton = await screen.findByRole('button', { name: /Activate current project/i });
        fireEvent.click(activateButton);

        expect(await screen.findByRole('heading', { name: /Activate project/i })).toBeInTheDocument();

        const confirmButton = screen.getByRole('button', { name: /^Activate$/i });
        fireEvent.click(confirmButton);

        await waitFor(() => {
            expect(updateRequestMade).toBe(true);
        });
    });
});
