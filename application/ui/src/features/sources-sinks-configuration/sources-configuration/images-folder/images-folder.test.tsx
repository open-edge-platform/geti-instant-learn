/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagesFolderSourceType, SourceCreateType, SourceUpdateType } from '@geti-prompt/api';
import { render } from '@geti-prompt/test-utils';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http, server } from '../../../../setup-test';
import { ImagesFolder } from './images-folder.component';

const getImagesFolderSource = (
    source: Partial<{ connected: boolean; imagesFolderPath: string }> = {}
): ImagesFolderSourceType => {
    return {
        id: '123',
        connected: source.connected ?? true,
        config: {
            seekable: true,
            images_folder_path: source.imagesFolderPath ?? '',
            source_type: 'images_folder',
        },
    };
};

class ImagesFolderSourcePage {
    constructor() {}

    get folderPathField() {
        return screen.getByRole('textbox', { name: 'Folder path' });
    }

    get applyButton() {
        return screen.getByRole('button', { name: 'Apply' });
    }

    async submit() {
        await userEvent.click(this.applyButton);
    }

    async setFolderPath(path: string) {
        await userEvent.clear(this.folderPathField);
        await userEvent.type(this.folderPathField, path);
    }
}

const renderImagesFolder = (source: ImagesFolderSourceType | undefined) => {
    const result = render(<ImagesFolder source={source} />);

    return {
        result,
        imagesFolderSourcePage: new ImagesFolderSourcePage(),
    };
};

describe('ImagesFolder', () => {
    it('disables submit button when path is empty', () => {
        const { imagesFolderSourcePage } = renderImagesFolder(undefined);

        expect(imagesFolderSourcePage.folderPathField).toHaveValue('');
        expect(imagesFolderSourcePage.applyButton).toBeDisabled();
    });

    it('disables submit button when path is equal to source path', () => {
        const source = getImagesFolderSource({ imagesFolderPath: '/path/to/folder' });

        const { imagesFolderSourcePage } = renderImagesFolder(source);

        expect(imagesFolderSourcePage.folderPathField).toHaveValue(source.config.images_folder_path);
        expect(imagesFolderSourcePage.applyButton).toBeDisabled();
    });

    it('enables submit button when path is different from source path', async () => {
        const imagesFolderPath = '/path/to/folder';
        const source = getImagesFolderSource({ imagesFolderPath: '/another/path' });

        const { imagesFolderSourcePage } = renderImagesFolder(source);

        expect(imagesFolderSourcePage.folderPathField).toHaveValue(source.config.images_folder_path);
        expect(imagesFolderSourcePage.applyButton).toBeDisabled();

        await imagesFolderSourcePage.setFolderPath(imagesFolderPath);

        expect(imagesFolderSourcePage.folderPathField).toHaveValue(imagesFolderPath);
        expect(imagesFolderSourcePage.applyButton).toBeEnabled();
    });

    it('creates source with provided path when source does not exist', async () => {
        const imagesFolderPath = '/path/to/folder';

        let body: SourceCreateType | null = null;
        let updateSourceCalled = false;

        server.use(
            http.post('/api/v1/projects/{project_id}/sources', async ({ request }) => {
                body = await request.json();

                return HttpResponse.json({}, { status: 201 });
            })
        );

        server.use(
            http.put('/api/v1/projects/{project_id}/sources/{source_id}', async () => {
                updateSourceCalled = true;

                return HttpResponse.json({}, { status: 200 });
            })
        );

        const { imagesFolderSourcePage } = renderImagesFolder(undefined);

        await imagesFolderSourcePage.setFolderPath(imagesFolderPath);
        await imagesFolderSourcePage.submit();

        await waitFor(() => {
            expect(updateSourceCalled).toBe(false);

            expect(body).toEqual(
                expect.objectContaining({
                    connected: true,
                    config: {
                        seekable: true,
                        source_type: 'images_folder',
                        images_folder_path: imagesFolderPath,
                    },
                })
            );
        });
    });

    it('updates source with provided path when source exists', async () => {
        const imagesFolderPath = '/path/to/folder';
        const source = getImagesFolderSource({ imagesFolderPath: '/another/path' });

        let body: SourceUpdateType | null = null;
        let createSourceCalled = false;

        server.use(
            http.post('/api/v1/projects/{project_id}/sources', async () => {
                createSourceCalled = true;
                return HttpResponse.json({}, { status: 201 });
            })
        );

        server.use(
            http.put('/api/v1/projects/{project_id}/sources/{source_id}', async ({ request }) => {
                body = await request.json();

                return HttpResponse.json({}, { status: 200 });
            })
        );

        const { imagesFolderSourcePage } = renderImagesFolder(source);

        await imagesFolderSourcePage.setFolderPath(imagesFolderPath);
        await imagesFolderSourcePage.submit();

        await waitFor(() => {
            expect(createSourceCalled).toBe(false);

            expect(body).toEqual(
                expect.objectContaining({
                    connected: true,
                    config: {
                        seekable: true,
                        source_type: 'images_folder',
                        images_folder_path: imagesFolderPath,
                    },
                })
            );
        });
    });
});
