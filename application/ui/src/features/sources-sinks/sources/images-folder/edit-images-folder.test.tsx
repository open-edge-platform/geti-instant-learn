/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagesFolderSourceType, SourceUpdateType } from '@geti-prompt/api';
import { getMockedImagesFolderSource, render } from '@geti-prompt/test-utils';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, vi } from 'vitest';

import { http, server } from '../../../../setup-test';
import { EditImagesFolder } from './edit-images-folder.component';

class EditImagesFolderSourcePage {
    constructor() {}

    get folderPathField() {
        return screen.getByRole('textbox', { name: 'Folder path' });
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
            await userEvent.click(this.saveAndConnectButton);
        }
    }

    async setFolderPath(path: string) {
        await userEvent.clear(this.folderPathField);
        await userEvent.type(this.folderPathField, path);
    }
}
const renderEditImagesFolder = ({
    source = getMockedImagesFolderSource(),
    onSaved = vi.fn(),
}: {
    source?: ImagesFolderSourceType;
    onSaved?: () => void;
} = {}) => {
    const result = render(<EditImagesFolder source={source} onSaved={onSaved} />);

    return {
        result,
        editImagesFolderSourcePage: new EditImagesFolderSourcePage(),
    };
};

describe('EditImagesFolder', () => {
    describe('Active source', () => {
        const activeSource = getMockedImagesFolderSource({ connected: true });

        it('displays only save button', () => {
            const { editImagesFolderSourcePage } = renderEditImagesFolder({ source: activeSource });

            expect(editImagesFolderSourcePage.saveButton).toBeInTheDocument();
            expect(editImagesFolderSourcePage.saveAndConnectButton).not.toBeInTheDocument();
        });

        it('disables save button when path is empty', () => {
            const { editImagesFolderSourcePage } = renderEditImagesFolder({ source: activeSource });

            expect(editImagesFolderSourcePage.folderPathField).toHaveValue('');
            expect(editImagesFolderSourcePage.saveButton).toBeDisabled();
        });

        it('disables save button when path is equal to source path', () => {
            const source = getMockedImagesFolderSource({ imagesFolderPath: '/path/to/folder', connected: true });

            const { editImagesFolderSourcePage } = renderEditImagesFolder({ source });

            expect(editImagesFolderSourcePage.folderPathField).toHaveValue(source.config.images_folder_path);
            expect(editImagesFolderSourcePage.saveButton).toBeDisabled();
        });

        it('enables submit button when path is different from source path', async () => {
            const imagesFolderPath = '/path/to/folder';
            const source = getMockedImagesFolderSource({ imagesFolderPath: '/another/path', connected: true });

            const { editImagesFolderSourcePage } = renderEditImagesFolder({ source });

            expect(editImagesFolderSourcePage.folderPathField).toHaveValue(source.config.images_folder_path);
            expect(editImagesFolderSourcePage.saveButton).toBeDisabled();

            await editImagesFolderSourcePage.setFolderPath(imagesFolderPath);

            expect(editImagesFolderSourcePage.folderPathField).toHaveValue(imagesFolderPath);
            expect(editImagesFolderSourcePage.saveButton).toBeEnabled();
        });

        it('updates source with provided path', async () => {
            const imagesFolderPath = '/path/to/folder';
            const source = getMockedImagesFolderSource({ imagesFolderPath: '/another/path', connected: true });

            let body: SourceUpdateType | null = null;

            server.use(
                http.put('/api/v1/projects/{project_id}/sources/{source_id}', async ({ request }) => {
                    body = await request.json();

                    return HttpResponse.json({}, { status: 200 });
                })
            );

            const { editImagesFolderSourcePage } = renderEditImagesFolder({ source });

            await editImagesFolderSourcePage.setFolderPath(imagesFolderPath);
            await editImagesFolderSourcePage.save();

            await waitFor(() => {
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

    describe('Inactive source', () => {
        const inactiveSource = getMockedImagesFolderSource({ connected: false });

        it('displays save and save&connect buttons', () => {
            const { editImagesFolderSourcePage } = renderEditImagesFolder({ source: inactiveSource });

            expect(editImagesFolderSourcePage.saveButton).toBeInTheDocument();
            expect(editImagesFolderSourcePage.saveAndConnectButton).toBeInTheDocument();
        });

        it('disables save buttons when path is empty', () => {
            const { editImagesFolderSourcePage } = renderEditImagesFolder({ source: inactiveSource });

            expect(editImagesFolderSourcePage.folderPathField).toHaveValue('');
            expect(editImagesFolderSourcePage.saveButton).toBeDisabled();
            expect(editImagesFolderSourcePage.saveAndConnectButton).toBeDisabled();
        });

        it('disables save buttons when path is equal to source path', () => {
            const source = getMockedImagesFolderSource({ imagesFolderPath: '/path/to/folder', connected: false });

            const { editImagesFolderSourcePage } = renderEditImagesFolder({ source });

            expect(editImagesFolderSourcePage.folderPathField).toHaveValue(source.config.images_folder_path);
            expect(editImagesFolderSourcePage.saveButton).toBeDisabled();
            expect(editImagesFolderSourcePage.saveAndConnectButton).toBeDisabled();
        });

        it('enables submit button when path is different from source path', async () => {
            const imagesFolderPath = '/path/to/folder';
            const source = getMockedImagesFolderSource({ imagesFolderPath: '/another/path', connected: false });

            const { editImagesFolderSourcePage } = renderEditImagesFolder({ source });

            expect(editImagesFolderSourcePage.folderPathField).toHaveValue(source.config.images_folder_path);
            expect(editImagesFolderSourcePage.saveButton).toBeDisabled();
            expect(editImagesFolderSourcePage.saveAndConnectButton).toBeDisabled();

            await editImagesFolderSourcePage.setFolderPath(imagesFolderPath);

            expect(editImagesFolderSourcePage.folderPathField).toHaveValue(imagesFolderPath);
            expect(editImagesFolderSourcePage.saveButton).toBeEnabled();
            expect(editImagesFolderSourcePage.saveAndConnectButton).toBeEnabled();
        });

        it('updates source with provided path', async () => {
            const imagesFolderPath = '/path/to/folder';
            const source = getMockedImagesFolderSource({ imagesFolderPath: '/another/path', connected: false });

            let body: SourceUpdateType | null = null;

            server.use(
                http.put('/api/v1/projects/{project_id}/sources/{source_id}', async ({ request }) => {
                    body = await request.json();

                    return HttpResponse.json({}, { status: 200 });
                })
            );

            const { editImagesFolderSourcePage } = renderEditImagesFolder({ source });

            await editImagesFolderSourcePage.setFolderPath(imagesFolderPath);
            await editImagesFolderSourcePage.save();

            await waitFor(() => {
                expect(body).toEqual(
                    expect.objectContaining({
                        connected: false,
                        config: {
                            seekable: true,
                            source_type: 'images_folder',
                            images_folder_path: imagesFolderPath,
                        },
                    })
                );
            });
        });

        it('updates source with provided path and connect', async () => {
            const imagesFolderPath = '/path/to/folder';
            const source = getMockedImagesFolderSource({ imagesFolderPath: '/another/path', connected: false });

            let body: SourceUpdateType | null = null;

            server.use(
                http.put('/api/v1/projects/{project_id}/sources/{source_id}', async ({ request }) => {
                    body = await request.json();

                    return HttpResponse.json({}, { status: 200 });
                })
            );

            const { editImagesFolderSourcePage } = renderEditImagesFolder({ source });

            await editImagesFolderSourcePage.setFolderPath(imagesFolderPath);
            await editImagesFolderSourcePage.saveAndConnect();

            await waitFor(() => {
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
});
