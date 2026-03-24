/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SampleDatasetSourceType } from '@/api';
import { render } from '@/test-utils';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { http, server } from '../../../../setup-test';
import { SampleDatasetCard } from './sample-dataset-card.component';

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

describe('SampleDatasetCard', () => {
    it('renders thumbnail, title and description from selected dataset id', async () => {
        server.use(
            http.get('/api/v1/system/datasets', () => {
                return HttpResponse.json(mockDatasetsResponse);
            })
        );

        const source = {
            id: 'source-id',
            active: false,
            config: {
                source_type: 'sample_dataset',
                seekable: true,
                dataset_id: DATASET_2.id,
            },
        } as SampleDatasetSourceType;

        render(<SampleDatasetCard source={source} menuItems={[]} onAction={vi.fn()} />);

        await screen.findByText('Nuts');
        expect(screen.getByText('Nuts dataset')).toBeVisible();

        const image = screen.getByRole('img', { name: 'Nuts' });
        expect(image).toHaveAttribute('src', DATASET_2.thumbnail);
    });
});
