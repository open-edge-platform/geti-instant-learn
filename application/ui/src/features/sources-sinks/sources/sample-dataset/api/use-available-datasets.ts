/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { baseUrl } from '@/api';
import { useQuery } from '@tanstack/react-query';

interface Dataset {
    id: string;
    name: string;
    description: string;
    thumbnail?: string | null;
}

interface DatasetsResponse {
    datasets: Dataset[];
}

const DATASETS_ENDPOINT = '/api/v1/system/datasets';

const fetchDatasets = async (): Promise<Dataset[]> => {
    const response = await fetch(`${baseUrl}${DATASETS_ENDPOINT}`);

    if (!response.ok) {
        throw new Error(`Unable to load datasets (${response.status})`);
    }

    const data = (await response.json()) as DatasetsResponse;
    return data.datasets;
};

export const useAvailableDatasets = () => {
    return useQuery({
        queryKey: ['available-datasets'],
        queryFn: fetchDatasets,
    });
};
