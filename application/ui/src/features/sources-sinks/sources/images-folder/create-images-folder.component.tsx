/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { Button } from '@geti/ui';

import { useCreateSource } from '../api/use-create-source';
import { ImagesFolderFields } from './images-folder-fields.component';
import { isFolderPathValid } from './utils';

interface CreateImagesFolderProps {
    onSaved: () => void;
}

export const CreateImagesFolder = ({ onSaved }: CreateImagesFolderProps) => {
    const [folderPath, setFolderPath] = useState<string>('');
    const createImagesFolderSource = useCreateSource();

    const isApplyDisabled = !isFolderPathValid(folderPath) || createImagesFolderSource.isPending;

    const submit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        createImagesFolderSource.mutate(
            {
                source_type: 'images_folder',
                images_folder_path: folderPath,
                seekable: true,
            },
            onSaved
        );
    };

    return (
        <form onSubmit={submit}>
            <ImagesFolderFields folderPath={folderPath} onSetFolderPath={setFolderPath} />

            <Button
                type={'submit'}
                variant={'accent'}
                isDisabled={isApplyDisabled}
                isPending={createImagesFolderSource.isPending}
                marginTop={'size-200'}
            >
                Apply
            </Button>
        </form>
    );
};
