/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { getMockedProject, getMockedSource, render } from '@geti-prompt/test-utils';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../setup-test';
import { MainContent } from './main-content.component';

describe('MainContent', () => {
    it('renders NotActiveProject if project is not active', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}', () => {
                return HttpResponse.json(
                    getMockedProject({
                        id: '1',
                        name: 'Inactive Project',
                        active: false,
                    })
                );
            })
        );

        render(<MainContent />);

        expect(await screen.findByText(/This project is set as inactive/i)).toBeInTheDocument();
    });

    it('renders NoSourcePlaceholder if there are no sources', async () => {
        // Mocks return no sources by default
        render(<MainContent />);

        expect(await screen.findByText(/Setup your input source/i)).toBeInTheDocument();
    });

    it('renders StreamContainer otherwise', async () => {
        http.get('/api/v1/projects/{project_id}/sources', () => {
            return HttpResponse.json({
                sources: [getMockedSource({ id: 'source-1' })],
            });
        });

        render(<MainContent />);

        expect(await screen.findByText(/Setup your input source/i)).toBeInTheDocument();
    });
});
