/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Suspense } from 'react';

import { SourceCreateType } from '@/api';
import { render } from '@/test-utils';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { http, server } from '../../../../setup-test';
import { CreateUsbCameraSource } from './create-usb-camera-source.component';

const USB_CAMERAS_RESPONSE = [
    { source_type: 'usb_camera' as const, device_id: 0, name: 'Webcam HD', seekable: false },
    { source_type: 'usb_camera' as const, device_id: 1, name: 'Webcam 4K', seekable: false },
];

const renderCreateUsbCameraSource = (onSaved = vi.fn()) => {
    server.use(
        http.get('/api/v1/system/source-types/{source_type}/sources', ({ response }) => {
            return response(200).json(USB_CAMERAS_RESPONSE);
        })
    );

    return render(
        <Suspense fallback={null}>
            <CreateUsbCameraSource onSaved={onSaved} />
        </Suspense>
    );
};

describe('CreateUsbCameraSource', () => {
    it('shows available cameras and submits the selected one', async () => {
        let body: SourceCreateType | null = null;
        const onSaved = vi.fn();

        server.use(
            http.post('/api/v1/projects/{project_id}/sources', async ({ request }) => {
                body = await request.json();
                return HttpResponse.json({}, { status: 201 });
            })
        );

        renderCreateUsbCameraSource(onSaved);

        expect(await screen.findByRole('button', { name: 'Webcam HD' })).toBeVisible();

        await userEvent.click(screen.getByRole('button', { name: 'Apply' }));

        await waitFor(() => {
            expect(body).toEqual(
                expect.objectContaining({
                    config: expect.objectContaining({
                        source_type: 'usb_camera',
                        device_id: 0,
                        name: 'Webcam HD',
                    }),
                })
            );
        });

        expect(onSaved).toHaveBeenCalled();
    });

    it('shows "No USB Cameras found" when no USB devices are available', async () => {
        server.use(
            http.get('/api/v1/system/source-types/{source_type}/sources', ({ response }) => {
                return response(200).json([]);
            })
        );

        render(
            <Suspense fallback={null}>
                <CreateUsbCameraSource onSaved={vi.fn()} />
            </Suspense>
        );

        expect(await screen.findByText('No USB Cameras found')).toBeVisible();
    });
});
