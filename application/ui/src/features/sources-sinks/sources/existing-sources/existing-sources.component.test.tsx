/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagesFolderSourceType, VideoFileSourceType } from '@/api';
import { getMockedImagesFolderSource, getMockedVideoFileSource, render } from '@/test-utils';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { http, server } from '../../../../setup-test';
import { ExistingSources } from './existing-sources.component';

const renderExistingSources = (
    sources: (VideoFileSourceType | ImagesFolderSourceType)[],
    onViewChange = vi.fn(),
    onSetSourceInEditionId = vi.fn()
) => {
    return render(
        <ExistingSources
            sources={sources}
            onViewChange={onViewChange}
            onSetSourceInEditionId={onSetSourceInEditionId}
        />
    );
};

describe('ExistingSources', () => {
    it('shows existing source title and parameters', async () => {
        const source = getMockedVideoFileSource({ filePath: '/home/user/video.mp4', active: true });

        renderExistingSources([source]);

        expect(await screen.findByText('Video file')).toBeInTheDocument();
        expect(screen.getByText('File path: /home/user/video.mp4')).toBeInTheDocument();
    });

    it('action menu for active source shows Edit and Delete but not Connect', async () => {
        const source = getMockedVideoFileSource({ active: true });

        renderExistingSources([source]);

        const card = await screen.findByTestId('pipeline-entity-card-video-file');
        await userEvent.click(within(card).getByRole('button'));

        expect(screen.getByRole('menuitem', { name: 'Edit' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Connect' })).not.toBeInTheDocument();
    });

    it('action menu for inactive source shows Connect, Edit, and Delete', async () => {
        const source = getMockedVideoFileSource({ active: false });

        renderExistingSources([source]);

        const card = await screen.findByTestId('pipeline-entity-card-video-file');
        await userEvent.click(within(card).getByRole('button'));

        expect(screen.getByRole('menuitem', { name: 'Connect' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Edit' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
    });

    it('does not call onViewChange when deleting one of multiple sources', async () => {
        const videoSource = getMockedVideoFileSource({ id: 'video-id', active: true });
        const imagesSource = getMockedImagesFolderSource({ active: true });
        const onViewChange = vi.fn();
        let deleted = false;

        server.use(
            http.delete('/api/v1/projects/{project_id}/sources/{source_id}', () => {
                deleted = true;
                return HttpResponse.json({}, { status: 204 });
            })
        );

        renderExistingSources([videoSource, imagesSource], onViewChange);

        const card = await screen.findByTestId('pipeline-entity-card-video-file');
        await userEvent.click(within(card).getByRole('button'));
        await userEvent.click(screen.getByRole('menuitem', { name: 'Delete' }));

        await waitFor(() => expect(deleted).toBe(true));

        expect(onViewChange).not.toHaveBeenCalled();
    });
});
