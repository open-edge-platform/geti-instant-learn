/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { ImagesFolderConfig, ImagesFolderSourceType } from '@geti-prompt/api';
import { Button, Content, ContextualHelp, Heading, Text, TextField } from '@geti/ui';

import { useCreateSource } from '../api/use-create-source';
import { useUpdateSource } from '../api/use-update-source';

interface ImagesFolderProps {
    source: ImagesFolderSourceType | undefined;
}

const FolderPathDescription = () => {
    return (
        <ContextualHelp variant='info'>
            <Heading>What is a folder path?</Heading>
            <Content>
                <Text>
                    A folder path is the location of a directory on your system.
                    <br />
                    Enter the absolute path (e.g. /Users/username/images) or relative path (e.g. ./data/images) to the
                    folder containing your images.
                </Text>
            </Content>
        </ContextualHelp>
    );
};

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

        const config: ImagesFolderConfig = {
            source_type: 'images_folder',
            images_folder_path: folderPath,
            seekable: true,
        };

        if (source === undefined) {
            createImagesFolderSource.mutate(config);
        } else {
            updateImagesFolderSource.mutate(source.id, config);
        }
    };

    return (
        <form onSubmit={submit}>
            <TextField
                label={'Folder path'}
                value={folderPath}
                onChange={setFolderPath}
                width={'100%'}
                contextualHelp={<FolderPathDescription />}
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
