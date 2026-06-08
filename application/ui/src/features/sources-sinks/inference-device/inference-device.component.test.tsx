/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { type DeviceInfoType } from '@/api';
import { ModelLoadingDialog } from '@/features/model-loading';
import { queryClient } from '@/query-client';
import { render } from '@/test-utils';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../../setup-test';
import { InferenceDevice } from './inference-device.component';

const renderComponent = (route = '/projects/1?mode=visual') =>
    render(<InferenceDevice />, { route, path: '/projects/:projectId' });

const mockDevices = (devices: DeviceInfoType[]) => {
    server.use(
        http.get('/api/v1/system/devices', () => {
            return HttpResponse.json(devices);
        })
    );
};

describe('InferenceDevice', () => {
    it('renders Auto plus all available devices', async () => {
        mockDevices([
            { type: 'cpu', name: 'Some CPU', memory: null, index: null },
            { type: 'cuda', name: 'NVIDIA GPU', memory: 16 * 1024 ** 3, index: 0 },
        ]);

        renderComponent();

        const picker = await screen.findByRole('button', { name: /inference device/i });
        fireEvent.click(picker);

        const listbox = await screen.findByRole('listbox');
        expect(within(listbox).getByRole('option', { name: 'Auto' })).toBeVisible();
        expect(within(listbox).getByRole('option', { name: 'Some CPU' })).toBeVisible();
        expect(within(listbox).getByRole('option', { name: 'NVIDIA GPU (16 GB)' })).toBeVisible();
    });

    it('appends [index] only when name+type collides', async () => {
        mockDevices([
            { type: 'xpu', name: 'Intel Arc', memory: 8 * 1024 ** 3, index: 0 },
            { type: 'xpu', name: 'Intel Arc', memory: 8 * 1024 ** 3, index: 1 },
            { type: 'cpu', name: 'Some CPU', memory: null, index: null },
        ]);

        renderComponent();

        const picker = await screen.findByRole('button', { name: /inference device/i });
        fireEvent.click(picker);

        const listbox = await screen.findByRole('listbox');
        expect(within(listbox).getByRole('option', { name: 'Intel Arc (8 GB) [0]' })).toBeVisible();
        expect(within(listbox).getByRole('option', { name: 'Intel Arc (8 GB) [1]' })).toBeVisible();
        expect(within(listbox).getByRole('option', { name: 'Some CPU' })).toBeVisible();
    });

    it('issues a PATCH with the selected key', async () => {
        mockDevices([
            { type: 'cpu', name: 'Some CPU', memory: null, index: null },
            { type: 'cuda', name: 'NVIDIA GPU', memory: null, index: 0 },
        ]);

        let updatePayload: unknown = null;
        server.use(
            http.put('/api/v1/projects/{project_id}', async ({ request }) => {
                updatePayload = await request.json();
                return HttpResponse.json({
                    id: '1',
                    name: 'Project #1',
                    active: true,
                    device: 'cuda-0',
                    prompt_mode: 'VISUAL',
                });
            })
        );

        renderComponent();

        const picker = await screen.findByRole('button', { name: /inference device/i });
        fireEvent.click(picker);
        fireEvent.click(await screen.findByRole('option', { name: 'NVIDIA GPU' }));

        await waitFor(() => {
            expect(updatePayload).toEqual({ device: 'cuda-0' });
        });
    });

    it('flags the model as loading after a device change so the blocking dialog can appear', async () => {
        mockDevices([
            { type: 'cpu', name: 'Some CPU', memory: null, index: null },
            { type: 'cuda', name: 'NVIDIA GPU', memory: null, index: 0 },
        ]);

        let modelStatusCalls = 0;
        server.use(
            http.put('/api/v1/projects/{project_id}', async () =>
                HttpResponse.json({
                    id: '1',
                    name: 'Project #1',
                    active: true,
                    device: 'cuda-0',
                    prompt_mode: 'VISUAL',
                })
            ),
            http.get('/api/v1/projects/{project_id}/model-status', () => {
                modelStatusCalls += 1;
                return HttpResponse.json({ status: 'ready' });
            })
        );

        renderComponent();

        const picker = await screen.findByRole('button', { name: /inference device/i });
        fireEvent.click(picker);
        fireEvent.click(await screen.findByRole('option', { name: 'NVIDIA GPU' }));

        await waitFor(() => {
            const cached = queryClient.getQueryData([
                'get',
                '/api/v1/projects/{project_id}/model-status',
                { params: { path: { project_id: '1' } } },
            ]);
            expect(cached).toEqual({ status: 'loading' });
        });
        expect(modelStatusCalls).toBe(0);
    });

    it('shows the blocking dialog after a device change', async () => {
        vi.useFakeTimers({ shouldAdvanceTime: true });

        mockDevices([
            { type: 'cpu', name: 'Some CPU', memory: null, index: null },
            { type: 'cuda', name: 'NVIDIA GPU', memory: null, index: 0 },
        ]);
        server.use(
            http.put('/api/v1/projects/{project_id}', async () =>
                HttpResponse.json({
                    id: '1',
                    name: 'Project #1',
                    active: true,
                    device: 'cuda-0',
                    prompt_mode: 'VISUAL',
                })
            )
        );

        render(
            <>
                <InferenceDevice />
                <ModelLoadingDialog />
            </>,
            { route: '/projects/1?mode=visual', path: '/projects/:projectId' }
        );

        const picker = await screen.findByRole('button', { name: /inference device/i });
        fireEvent.click(picker);
        fireEvent.click(await screen.findByRole('option', { name: 'NVIDIA GPU' }));

        await act(async () => {
            await vi.advanceTimersByTimeAsync(500);
        });

        await waitFor(() => {
            expect(screen.getByRole('dialog', { name: /loading model/i })).toBeVisible();
        });

        vi.useRealTimers();
    });
});
