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

const emptyDatasetsResponse = {
    datasets: [],
    pagination: {
        count: 0,
        total: 0,
        offset: 0,
        limit: 20,
    },
};

const renderCreateSampleDataset = (onSaved = vi.fn(), datasetsResponse = mockDatasetsResponse) => {
    server.use(
        http.get('/api/v1/system/datasets', () => {
            return HttpResponse.json(datasetsResponse);
        })
    );

    return render(<CreateSampleDataset onSaved={onSaved} />);
};

describe('CreateSampleDataset', () => {
    it('renders first dataset thumbnail and metadata by default', async () => {
        renderCreateSampleDataset();

        expect(await screen.findByRole('heading', { name: 'Aquarium' })).toBeVisible();
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

        expect(await screen.findByRole('heading', { name: 'Aquarium' })).toBeVisible();

        fireEvent.click(screen.getByRole('button', { name: 'Apply' }));

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

    it('updates preview and submit payload when a different dataset is selected', async () => {
        let body: SourceCreateType | null = null;

        server.use(
            http.get('/api/v1/system/datasets', () => {
                return HttpResponse.json(mockDatasetsResponse);
            }),
            http.post('/api/v1/projects/{project_id}/sources', async ({ request }) => {
                body = await request.json();
                return HttpResponse.json({}, { status: 201 });
            })
        );

        render(<CreateSampleDataset onSaved={vi.fn()} />);

        expect(await screen.findByRole('heading', { name: 'Aquarium' })).toBeVisible();

        fireEvent.click(screen.getByLabelText('Dataset'));
        fireEvent.click(screen.getByRole('option', { name: DATASET_2.name }));

        expect(await screen.findByRole('heading', { name: 'Nuts' })).toBeVisible();

        const image = screen.getByRole('img', { name: 'Nuts' });
        expect(image).toHaveAttribute('src', DATASET_2.thumbnail);

        fireEvent.click(screen.getByRole('button', { name: 'Apply' }));

        await waitFor(() => {
            expect(body).toEqual(
                expect.objectContaining({
                    active: true,
                    config: {
                        seekable: true,
                        source_type: 'sample_dataset',
                        dataset_id: DATASET_2.id,
                    },
                })
            );
        });
    });

    it('renders empty datasets message and keeps apply disabled when no datasets are available', async () => {
        renderCreateSampleDataset(vi.fn(), emptyDatasetsResponse);

        expect(await screen.findByText('No sample datasets are available.')).toBeVisible();
        expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
    });
});
