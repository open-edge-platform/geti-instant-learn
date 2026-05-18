/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { usePromptMode } from '@/hooks';
import { render } from '@/test-utils';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../../setup-test';
import { PromptModes } from './prompt-modes.component';

const mockProjectWithMode = (promptMode: 'TEXT' | 'VISUAL') => {
    server.use(
        http.get('/api/v1/projects/{project_id}', () => {
            return HttpResponse.json({
                id: '1',
                name: 'Project #1',
                active: true,
                device: 'cpu',
                prompt_mode: promptMode,
            });
        })
    );
};

describe('PromptModes', () => {
    it('renders prompt mode toggle buttons', async () => {
        render(<PromptModes />);

        expect(await screen.findByText('Prompt Mode')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Visual Prompt' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Text Prompt' })).toBeInTheDocument();
    });

    it('defaults to visual mode from project data', async () => {
        render(<PromptModes />);

        const visualButton = await screen.findByRole('button', { name: 'Visual Prompt' });
        expect(visualButton).toHaveAttribute('aria-pressed', 'true');
    });

    it('shows text mode as selected when project prompt_mode is TEXT', async () => {
        mockProjectWithMode('TEXT');
        render(<PromptModes />);

        const textButton = await screen.findByRole('button', { name: 'Text Prompt' });
        expect(textButton).toHaveAttribute('aria-pressed', 'true');
    });

    it('shows visual mode as selected when project prompt_mode is VISUAL', async () => {
        mockProjectWithMode('VISUAL');
        render(<PromptModes />);

        const visualButton = await screen.findByRole('button', { name: 'Visual Prompt' });
        expect(visualButton).toHaveAttribute('aria-pressed', 'true');
    });

    it('calls PUT to update project when switching mode', async () => {
        let capturedBody: Record<string, unknown> | undefined;

        server.use(
            http.put('/api/v1/projects/{project_id}', async ({ request }) => {
                capturedBody = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json({
                    id: '1',
                    name: 'Project #1',
                    active: true,
                    device: 'cpu',
                    prompt_mode: 'TEXT',
                });
            })
        );

        render(<PromptModes />);

        const textButton = await screen.findByRole('button', { name: 'Text Prompt' });
        fireEvent.click(textButton);

        await waitFor(() => {
            expect(capturedBody).toEqual(expect.objectContaining({ prompt_mode: 'TEXT' }));
        });
    });
});

describe('usePromptMode', () => {
    const TestComponent = () => {
        const [mode] = usePromptMode();

        return <div aria-label='mode'>{mode}</div>;
    };

    it('returns VISUAL when project prompt_mode is VISUAL', async () => {
        render(<TestComponent />);

        expect(await screen.findByLabelText('mode')).toHaveTextContent('VISUAL');
    });

    it('returns TEXT when project prompt_mode is TEXT', async () => {
        mockProjectWithMode('TEXT');
        render(<TestComponent />);

        expect(await screen.findByLabelText('mode')).toHaveTextContent('TEXT');
    });
});
