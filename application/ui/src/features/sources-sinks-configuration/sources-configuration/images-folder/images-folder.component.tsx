/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { ImagesFolderSourceType } from '@geti-prompt/api';
import { Folder } from '@geti-prompt/icons';
import { ActionButton, Button, Flex, TextField } from '@geti/ui';

import { useCreateSource } from '../hooks/use-create-source';
import { useUpdateSource } from '../hooks/use-update-source';

import styles from './images-folder.module.scss';

interface ImagesFolderProps {
    source: ImagesFolderSourceType | undefined;
}

export const ImagesFolder = ({ source }: ImagesFolderProps) => {
    const [folderPath, setFolderPath] = useState(source?.config?.images_folder_path ?? '');
    const createImagesFolderSource = useCreateSource();
    const updateImagesFolderSource = useUpdateSource();

    const isApplyDisabled =
        folderPath.trim().length === 0 ||
        createImagesFolderSource.isPending ||
        updateImagesFolderSource.isPending ||
        (folderPath === source?.config?.images_folder_path && source?.connected);

    const showDirectoryPicker = async () => {
        if (window.showDirectoryPicker === undefined) return;

        const handle = await window.showDirectoryPicker();

        setFolderPath(handle.name);
    };

    const submit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (source !== undefined) {
            updateImagesFolderSource.mutate(source.id, {
                source_type: 'images_folder',
                images_folder_path: folderPath,
            });

            return;
        }

        createImagesFolderSource.mutate({
            source_type: 'images_folder',
            images_folder_path: folderPath,
        });
    };

    return (
        <form onSubmit={submit}>
            <Flex alignItems={'end'} gap={'size-50'}>
                <TextField label={'Folder path'} value={folderPath} onChange={setFolderPath} flex={1} />
                <ActionButton
                    isQuiet
                    aria-label={'Select folder'}
                    UNSAFE_className={styles.imagesFolderButton}
                    onPress={showDirectoryPicker}
                >
                    <Folder />
                </ActionButton>
            </Flex>

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
