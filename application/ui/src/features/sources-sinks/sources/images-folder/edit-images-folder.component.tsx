/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { ImagesFolderSourceType } from '@geti-prompt/api';
import { Flex } from '@geti/ui';

import { useUpdateSource } from '../api/use-update-source';
import { EditSourceButtons } from '../edit-sources/edit-source-buttons.component';
import { ImagesFolderFields } from './images-folder-fields.component';
import { isFolderPathValid } from './utils';

interface EditImagesFolderProps {
    source: ImagesFolderSourceType;
    onSaved: () => void;
}

export const EditImagesFolder = ({ source, onSaved }: EditImagesFolderProps) => {
    const [folderPath, setFolderPath] = useState<string>(source.config.images_folder_path);

    const updateImagesFolderSource = useUpdateSource();
    const isActiveSource = source.connected;

    const isButtonDisabled =
        !isFolderPathValid(folderPath) ||
        folderPath === source.config.images_folder_path ||
        updateImagesFolderSource.isPending;

    const handleUpdateImagesFolder = (connected: boolean) => {
        updateImagesFolderSource.mutate(
            source.id,
            {
                config: {
                    source_type: 'images_folder',
                    images_folder_path: folderPath,
                    seekable: true,
                },
                connected,
            },
            onSaved
        );
    };

    const handleSave = () => {
        handleUpdateImagesFolder(false);
    };

    const handleSaveAndConnect = () => {
        handleUpdateImagesFolder(true);
    };

    return (
        <Flex direction={'column'} gap={'size-200'}>
            <ImagesFolderFields folderPath={folderPath} onSetFolderPath={setFolderPath} />

            <EditSourceButtons
                isActiveSource={isActiveSource}
                onSave={handleSave}
                onSaveAndConnect={handleSaveAndConnect}
                isDisabled={isButtonDisabled}
                isPending={updateImagesFolderSource.isPending}
            />
        </Flex>
    );
};
