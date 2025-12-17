/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { ImagesFolderSourceType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Flex, Form } from '@geti/ui';
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
        { folderPath, active }: { folderPath: string; active: boolean },
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
                active,
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
    const isActiveSource = source.active;

    const isButtonDisabled =
        !isFolderPathValid(folderPath) ||
        folderPath === source.config.images_folder_path ||
        updateImagesFolderSource.isPending;

    const handleUpdateImagesFolder = (active: boolean) => {
        updateImagesFolderSource.mutate({ folderPath, active }, onSaved);
    };

    const handleSave = () => {
        handleUpdateImagesFolder(source.active);
    };

    const handleSaveAndConnect = () => {
        handleUpdateImagesFolder(true);
    };

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        handleSave();
    };

    return (
        <Form validationBehavior={'native'} onSubmit={handleSubmit}>
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
        </Form>
    );
};
