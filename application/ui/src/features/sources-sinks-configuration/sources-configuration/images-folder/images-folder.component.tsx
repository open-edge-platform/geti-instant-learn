/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { ImagesFolderSourceType } from '@geti-prompt/api';
import { Button, TextField } from '@geti/ui';

import { useCreateSource } from '../hooks/use-create-source';
import { useUpdateSource } from '../hooks/use-update-source';

interface ImagesFolderProps {
    source: ImagesFolderSourceType | undefined;
}

export const ImagesFolder = ({ source }: ImagesFolderProps) => {
    const [folderPath, setFolderPath] = useState(source?.config?.images_folder_path ?? '');
    const createImagesFolderSource = useCreateSource();
    const updateImagesFolderSource = useUpdateSource();

    const isFolderPathValid = folderPath.trim().length > 0;

    const isApplyDisabled =
        !isFolderPathValid ||
        createImagesFolderSource.isPending ||
        updateImagesFolderSource.isPending ||
        (folderPath === source?.config?.images_folder_path && source?.connected);

    const submit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (source === undefined) {
            createImagesFolderSource.mutate({
                source_type: 'images_folder',
                images_folder_path: folderPath,
                seekable: true,
            });
        } else {
            updateImagesFolderSource.mutate(source.id, {
                source_type: 'images_folder',
                images_folder_path: folderPath,
                seekable: true,
            });
        }
    };

    return (
        <form onSubmit={submit}>
            <TextField
                label={'Folder path'}
                value={folderPath}
                onChange={setFolderPath}
                width={'100%'}
                description={
                    isFolderPathValid
                        ? undefined
                        : // eslint-disable-next-line max-len
                          'Enter the absolute or relative path to the folder containing images (e.g., /home/user/images or ./data/images)'
                }
            />

            <Button
                type={'submit'}
                variant={'accent'}
                isDisabled={isApplyDisabled}
                isPending={createImagesFolderSource.isPending || updateImagesFolderSource.isPending}
                marginTop={'size-200'}
            >
                Apply
            </Button>
        </form>
    );
};
