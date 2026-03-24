/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SourceCreateType } from '@/api';
import { render } from '@/test-utils';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { http, server } from '../../../../setup-test';
import { CreateSampleDataset } from './create-sample-dataset.component';

const DATASET_1 = {
    id: '11111111-1111-1111-1111-111111111111',
    name: 'Aquarium',
    description: 'Aquarium dataset',
    thumbnail: 'data:image/jpeg;base64,AAAA',
};

const DATASET_2 = {
    id: '22222222-2222-2222-2222-222222222222',
    name: 'Nuts',
    description: 'Nuts dataset',
    thumbnail: 'data:image/jpeg;base64,BBBB',
};

const mockDatasetsResponse = {
    datasets: [DATASET_1, DATASET_2],
    pagination: {
        count: 2,
        total: 2,
        offset: 0,
        limit: 20,
    },
};

const renderCreateSampleDataset = (onSaved = vi.fn()) => {
    server.use(
        http.get('/api/v1/system/datasets', () => {
            return HttpResponse.json(mockDatasetsResponse);
        })
    );

    return render(<CreateSampleDataset onSaved={onSaved} />);
};

describe('CreateSampleDataset', () => {
    it('renders first dataset thumbnail and metadata by default', async () => {
        renderCreateSampleDataset();

        await screen.findByRole('heading', { name: 'Aquarium' });
        expect(screen.getByText('Aquarium dataset')).toBeVisible();

        const image = screen.getByRole('img', { name: 'Aquarium' });
        expect(image).toHaveAttribute('src', DATASET_1.thumbnail);
    });

    it('submits sample dataset source with selected default dataset_id', async () => {
        let body: SourceCreateType | null = null;
        const onSaved = vi.fn();

        server.use(
            http.get('/api/v1/system/datasets', () => {
                return HttpResponse.json(mockDatasetsResponse);
            }),
            http.post('/api/v1/projects/{project_id}/sources', async ({ request }) => {
                body = await request.json();
                return HttpResponse.json({}, { status: 201 });
            })
        );

        render(<CreateSampleDataset onSaved={onSaved} />);

        await screen.findByRole('heading', { name: 'Aquarium' });

        const applyButton = screen.getByRole('button', { name: 'Apply' });
        const form = applyButton.closest('form');
        expect(form).not.toBeNull();
        fireEvent.submit(form as HTMLFormElement);

        await waitFor(() => {
            expect(body).toEqual(
                expect.objectContaining({
                    active: true,
                    config: {
                        seekable: true,
                        source_type: 'sample_dataset',
                        dataset_id: DATASET_1.id,
                    },
                })
            );
        });

        expect(onSaved).toHaveBeenCalled();
    });
});
