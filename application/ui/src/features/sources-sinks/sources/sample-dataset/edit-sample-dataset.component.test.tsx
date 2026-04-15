/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SampleDatasetSourceType, SourceUpdateType } from '@/api';
import { getMockedSampleDatasetSource, render } from '@/test-utils';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, vi } from 'vitest';

import { http, server } from '../../../../setup-test';
import { EditSampleDataset } from './edit-sample-dataset.component';

const DATASET_1 = {
    id: '11111111-1111-1111-1111-111111111111',
    name: 'Aquarium',
    thumbnail: 'data:image/jpeg;base64,AAAA',
};

const DATASET_2 = {
    id: '22222222-2222-2222-2222-222222222222',
    name: 'Nuts',
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

class EditSampleDatasetPage {
    constructor() {}

    get datasetPicker() {
        return screen.getByRole('button', { name: /Dataset/i });
    }

    get saveButton() {
        return screen.getByRole('button', { name: 'Save' });
    }

    get saveAndConnectButton() {
        return screen.queryByRole('button', { name: 'Save & Connect' });
    }

    async save() {
        await userEvent.click(this.saveButton);
    }

    async saveAndConnect() {
        const btn = this.saveAndConnectButton;
        if (btn) {
            await userEvent.click(btn);
        }
    }

    async selectDataset(datasetName: string) {
        await userEvent.click(this.datasetPicker);
        await userEvent.click(screen.getByRole('option', { name: datasetName }));
    }
}

const renderEditSampleDataset = ({
    source = getMockedSampleDatasetSource(),
    onSaved = vi.fn(),
    datasetsResponse = mockDatasetsResponse,
}: {
    source?: SampleDatasetSourceType;
    onSaved?: () => void;
    datasetsResponse?: typeof mockDatasetsResponse | typeof emptyDatasetsResponse;
} = {}) => {
    server.use(
        http.get('/api/v1/system/datasets', () => {
            return HttpResponse.json(datasetsResponse);
        })
    );

    const result = render(<EditSampleDataset source={source} onSaved={onSaved} />);

    return {
        result,
        editSampleDatasetPage: new EditSampleDatasetPage(),
    };
};

describe('EditSampleDataset', () => {
    describe('Active source', () => {
        const activeSource = getMockedSampleDatasetSource({ active: true, datasetId: DATASET_1.id });

        it('displays only save button', async () => {
            const { editSampleDatasetPage } = renderEditSampleDataset({ source: activeSource });

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_1.name })).toBeVisible();
            });

            expect(editSampleDatasetPage.saveButton).toBeInTheDocument();
            expect(editSampleDatasetPage.saveAndConnectButton).not.toBeInTheDocument();
        });

        it('displays current dataset name and thumbnail', async () => {
            renderEditSampleDataset({ source: activeSource });

            expect(await screen.findByRole('heading', { name: DATASET_1.name })).toBeVisible();

            const image = screen.getByRole('img', { name: DATASET_1.name });
            expect(image).toHaveAttribute('src', DATASET_1.thumbnail);
        });

        it('disables save button when dataset is not changed', async () => {
            const { editSampleDatasetPage } = renderEditSampleDataset({ source: activeSource });

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_1.name })).toBeVisible();
            });

            expect(editSampleDatasetPage.saveButton).toBeDisabled();
        });

        it('enables save button when a different dataset is selected', async () => {
            const { editSampleDatasetPage } = renderEditSampleDataset({ source: activeSource });

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_1.name })).toBeVisible();
            });

            expect(editSampleDatasetPage.saveButton).toBeDisabled();

            await editSampleDatasetPage.selectDataset(DATASET_2.name);

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_2.name })).toBeVisible();
            });

            expect(editSampleDatasetPage.saveButton).toBeEnabled();
        });

        it('updates source with selected dataset', async () => {
            const { editSampleDatasetPage } = renderEditSampleDataset({ source: activeSource });

            let body: SourceUpdateType | null = null;

            server.use(
                http.put('/api/v1/projects/{project_id}/sources/{source_id}', async ({ request }) => {
                    body = await request.json();
                    return HttpResponse.json({}, { status: 200 });
                })
            );

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_1.name })).toBeVisible();
            });

            await editSampleDatasetPage.selectDataset(DATASET_2.name);
            await editSampleDatasetPage.save();

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

        it('updates preview when dataset selection changes', async () => {
            const { editSampleDatasetPage } = renderEditSampleDataset({ source: activeSource });

            expect(await screen.findByRole('heading', { name: DATASET_1.name })).toBeVisible();

            let image = screen.getByRole('img', { name: DATASET_1.name });
            expect(image).toHaveAttribute('src', DATASET_1.thumbnail);

            await editSampleDatasetPage.selectDataset(DATASET_2.name);

            expect(await screen.findByRole('heading', { name: DATASET_2.name })).toBeVisible();

            image = screen.getByRole('img', { name: DATASET_2.name });
            expect(image).toHaveAttribute('src', DATASET_2.thumbnail);
        });
    });

    describe('Inactive source', () => {
        const inactiveSource = getMockedSampleDatasetSource({ active: false, datasetId: DATASET_1.id });

        it('displays save and save&connect buttons', async () => {
            const { editSampleDatasetPage } = renderEditSampleDataset({ source: inactiveSource });

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_1.name })).toBeVisible();
            });

            expect(editSampleDatasetPage.saveButton).toBeInTheDocument();
            expect(editSampleDatasetPage.saveAndConnectButton).toBeInTheDocument();
        });

        it('disables save buttons when dataset is not changed', async () => {
            const { editSampleDatasetPage } = renderEditSampleDataset({ source: inactiveSource });

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_1.name })).toBeVisible();
            });

            expect(editSampleDatasetPage.saveButton).toBeDisabled();
            expect(editSampleDatasetPage.saveAndConnectButton).toBeDisabled();
        });

        it('enables save buttons when a different dataset is selected', async () => {
            const { editSampleDatasetPage } = renderEditSampleDataset({ source: inactiveSource });

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_1.name })).toBeVisible();
            });

            expect(editSampleDatasetPage.saveButton).toBeDisabled();
            expect(editSampleDatasetPage.saveAndConnectButton).toBeDisabled();

            await editSampleDatasetPage.selectDataset(DATASET_2.name);

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_2.name })).toBeVisible();
            });

            expect(editSampleDatasetPage.saveButton).toBeEnabled();
            expect(editSampleDatasetPage.saveAndConnectButton).toBeEnabled();
        });

        it('updates source with selected dataset', async () => {
            const { editSampleDatasetPage } = renderEditSampleDataset({ source: inactiveSource });

            let body: SourceUpdateType | null = null;

            server.use(
                http.put('/api/v1/projects/{project_id}/sources/{source_id}', async ({ request }) => {
                    body = await request.json();
                    return HttpResponse.json({}, { status: 200 });
                })
            );

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_1.name })).toBeVisible();
            });

            await editSampleDatasetPage.selectDataset(DATASET_2.name);
            await editSampleDatasetPage.save();

            await waitFor(() => {
                expect(body).toEqual(
                    expect.objectContaining({
                        active: false,
                        config: {
                            seekable: true,
                            source_type: 'sample_dataset',
                            dataset_id: DATASET_2.id,
                        },
                    })
                );
            });
        });

        it('updates source and activates it when save&connect is clicked', async () => {
            const { editSampleDatasetPage } = renderEditSampleDataset({ source: inactiveSource });

            let body: SourceUpdateType | null = null;

            server.use(
                http.put('/api/v1/projects/{project_id}/sources/{source_id}', async ({ request }) => {
                    body = await request.json();
                    return HttpResponse.json({}, { status: 200 });
                })
            );

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_1.name })).toBeVisible();
            });

            await editSampleDatasetPage.selectDataset(DATASET_2.name);
            await editSampleDatasetPage.saveAndConnect();

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
    });

    describe('Empty datasets', () => {
        it('displays message when no datasets are available', async () => {
            renderEditSampleDataset({ datasetsResponse: emptyDatasetsResponse });

            expect(await screen.findByText('No sample datasets are available.')).toBeVisible();
        });

        it('disables save button when no datasets are available', async () => {
            const { editSampleDatasetPage } = renderEditSampleDataset({
                datasetsResponse: emptyDatasetsResponse,
            });

            await waitFor(() => {
                expect(screen.getByText('No sample datasets are available.')).toBeVisible();
            });

            expect(editSampleDatasetPage.saveButton).toBeDisabled();
        });
    });

    describe('Callbacks', () => {
        it('calls onSaved callback after successful save', async () => {
            const onSaved = vi.fn();
            const source = getMockedSampleDatasetSource({ datasetId: DATASET_1.id });
            const { editSampleDatasetPage } = renderEditSampleDataset({ source, onSaved });

            server.use(
                http.put('/api/v1/projects/{project_id}/sources/{source_id}', async () => {
                    return HttpResponse.json({}, { status: 200 });
                })
            );

            await waitFor(() => {
                expect(screen.getByRole('heading', { name: DATASET_1.name })).toBeVisible();
            });

            await editSampleDatasetPage.selectDataset(DATASET_2.name);
            await editSampleDatasetPage.save();

            await waitFor(() => {
                expect(onSaved).toHaveBeenCalled();
            });
        });
    });
});
