/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { ImagesFolderSourceType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Flex } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';

import { useUpdateSource } from '../api/use-update-source';
import { EditSourceButtons } from '../edit-sources/edit-source-buttons.component';
import { ImagesFolderFields } from './images-folder-fields.component';
import { isFolderPathValid } from './utils';

interface EditImagesFolderProps {
    source: ImagesFolderSourceType;
    onSaved: () => void;
}

const useUpdateImagesFolderSource = (sourceId: string) => {
    const updateSource = useUpdateSource();
    const queryClient = useQueryClient();
    const { projectId } = useProjectIdentifier();

    const updateImagesFolderSource = (
        { folderPath, connected }: { folderPath: string; connected: boolean },
        onSuccess: () => void
    ) => {
        updateSource.mutate(
            sourceId,
            {
                config: {
                    source_type: 'images_folder',
                    seekable: true,
                    images_folder_path: folderPath,
                },
                connected,
            },
            () => {
                const params = {
                    path: {
                        project_id: projectId,
                        source_id: sourceId,
                    },
                };

                queryClient.invalidateQueries({
                    queryKey: [
                        'get',
                        '/api/v1/projects/{project_id}/sources/{source_id}/frames',
                        {
                            params,
                        },
                    ],
                });

                queryClient.invalidateQueries({
                    queryKey: [
                        'get',
                        '/api/v1/projects/{project_id}/sources/{source_id}/frames/index',
                        {
                            params,
                        },
                    ],
                });

                onSuccess();
            }
        );
    };

    return {
        mutate: updateImagesFolderSource,
        isPending: updateSource.isPending,
    };
};

export const EditImagesFolder = ({ source, onSaved }: EditImagesFolderProps) => {
    const [folderPath, setFolderPath] = useState<string>(source.config.images_folder_path);

    const updateImagesFolderSource = useUpdateImagesFolderSource(source.id);
    const isActiveSource = source.connected;

    const isButtonDisabled =
        !isFolderPathValid(folderPath) ||
        folderPath === source.config.images_folder_path ||
        updateImagesFolderSource.isPending;

    const handleUpdateImagesFolder = (connected: boolean) => {
        updateImagesFolderSource.mutate({ folderPath, connected }, onSaved);
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
