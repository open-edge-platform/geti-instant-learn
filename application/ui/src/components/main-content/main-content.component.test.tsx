/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { getMockedSource, render } from '@geti-prompt/test-utils';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { WebRTCConnectionProvider } from 'src/features/stream/web-rtc/web-rtc-connection-provider';

import { http, server } from '../../setup-test';
import { MainContent } from './main-content.component';

describe('MainContent', () => {
    it('renders NoSourcePlaceholder if there are no sources', async () => {
        // Mocks return no sources by default
        render(<MainContent />);

        expect(await screen.findByText(/Setup your input source/i)).toBeInTheDocument();
    });

    it('renders StreamContainer otherwise', async () => {
        server.use(
            http.get('/api/v1/projects/{project_id}/sources', () => {
                return HttpResponse.json({
                    sources: [getMockedSource({ id: 'source-1' })],
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
            <WebRTCConnectionProvider>
                <MainContent />
            </WebRTCConnectionProvider>
        );

        expect(await screen.findByLabelText('Start stream')).toBeInTheDocument();
    });
});
