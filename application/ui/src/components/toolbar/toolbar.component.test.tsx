/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { getMockedSource, render } from '@/test-utils';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { StreamConnectionProvider } from '../../features/stream/mjpeg/stream-connection-provider';
import { http, server } from '../../setup-test';
import { Toolbar } from './toolbar.component';

describe('Toolbar', () => {
    it('does not render stream status when there are no sources', async () => {
        // Default mock returns empty sources array
        render(
            <StreamConnectionProvider>
                <Toolbar />
            </StreamConnectionProvider>
        );

        expect(screen.queryByRole('status')).not.toBeInTheDocument();
    });

    it('renders status when there are sources', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}/sources', () => {
                return HttpResponse.json({
                    sources: [
                        getMockedSource({
                            id: 'source-1',
                            active: true,
                        }),
                    ],
                    pagination: {
                        count: 1,
                        total: 1,
                        limit: 10,
                        offset: 0,
                    },
                });
            })
        );

        render(
            <StreamConnectionProvider>
                <Toolbar />
            </StreamConnectionProvider>
        );

        expect(await screen.findByRole('status')).toBeInTheDocument();
    });

    it('always renders SourcesSinks button', async () => {
        render(
            <StreamConnectionProvider>
                <Toolbar />
            </StreamConnectionProvider>
        );

        expect(await screen.findByRole('button', { name: 'Pipeline configuration' })).toBeInTheDocument();
    });
});
