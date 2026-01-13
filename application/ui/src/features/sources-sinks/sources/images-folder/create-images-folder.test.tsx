/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SourceCreateType } from '@geti-prompt/api';
import { render } from '@geti-prompt/test-utils';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { http, server } from '../../../../setup-test';
import { CreateImagesFolder } from './create-images-folder.component';

class ImagesFolderSourcePage {
    constructor() {}

    get folderPathField() {
        return screen.getByRole('textbox', { name: /Folder path/ });
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

const renderImagesFolder = (onSaved = vi.fn()) => {
    const result = render(<CreateImagesFolder onSaved={onSaved} />);

    return {
        result,
        imagesFolderSourcePage: new ImagesFolderSourcePage(),
    };
};

describe('CreateImagesFolder', () => {
    it('disables submit button when path is empty', () => {
        const { imagesFolderSourcePage } = renderImagesFolder();

        expect(imagesFolderSourcePage.folderPathField).toHaveValue('');
        expect(imagesFolderSourcePage.applyButton).toBeDisabled();
    });

    it('creates source with provided path', async () => {
        const imagesFolderPath = '/path/to/folder';

        let body: SourceCreateType | null = null;

        server.use(
            http.post('/api/v1/projects/{project_id}/sources', async ({ request }) => {
                body = await request.json();

                return HttpResponse.json({}, { status: 201 });
            })
        );

        const onSaved = vi.fn();
        const { imagesFolderSourcePage } = renderImagesFolder(onSaved);

        await imagesFolderSourcePage.setFolderPath(imagesFolderPath);

        expect(imagesFolderSourcePage.applyButton).toBeEnabled();

        await imagesFolderSourcePage.submit();

        await waitFor(() => {
            expect(body).toEqual(
                expect.objectContaining({
                    active: true,
                    config: {
                        seekable: true,
                        source_type: 'images_folder',
                        images_folder_path: imagesFolderPath,
                    },
                })
            );
        });

        expect(onSaved).toHaveBeenCalled();
    });
});
