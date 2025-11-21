/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Source } from '@geti-prompt/api';
import { useCurrentProject } from '@geti-prompt/hooks';
import { Flex } from '@geti/ui';

import { AddNewSource } from './add-new-source/add-new-source.component';
import { useDeleteSource } from './api/use-delete-source';
import { useUpdateSource } from './api/use-update-source';
import { ImagesFolderSourceReadOnly } from './images-folder/images-folder-read-only.component';
import { isImagesFolderSource, isWebcamSource, SourcesViews } from './utils';
import { WebcamSourceReadOnly } from './webcam/webcam-source-read-only.component';

interface ExistingSourcesProps {
    onViewChange: (view: SourcesViews) => void;
    sources: Source[];
    activeSource: Source | undefined;
    onSetSourceInEditionId: (sourceId: string) => void;
}

const getMenuItems = ({ isActiveProject, isActiveSource }: { isActiveProject: boolean; isActiveSource: boolean }) => {
    const items = [
        {
            key: 'connect',
            label: 'Connect',
        },
        {
            key: 'edit',
            label: 'Edit',
        },
        {
            key: 'delete',
            label: 'Delete',
        },
    ];

    return items.filter((item) => {
        if (item.key === 'connect' && isActiveSource) {
            return false;
        }
        if (item.key === 'edit' && !isActiveProject) {
            return false;
        }
        return true;
    });
};

export const ExistingSourcesList = ({
    sources,
    onViewChange,
    activeSource,
    onSetSourceInEditionId,
}: ExistingSourcesProps) => {
    const { data: project } = useCurrentProject();
    const isActiveProject = project.active;

    const updateSource = useUpdateSource();
    const deleteSource = useDeleteSource();

    const handleAction = (source: Source) => (action: string) => {
        if (action === 'edit') {
            onViewChange('edit');
            onSetSourceInEditionId(source.id);
        } else if (action === 'connect') {
            updateSource.mutate(source.id, {
                config: source.config,
                connected: true,
            });
        } else if (action === 'delete') {
            deleteSource.mutate({ params: { path: { project_id: project.id, source_id: source.id } } });
        }
    };

    return (
        <Flex direction={'column'} gap={'size-200'}>
            <AddNewSource onAddNewSource={() => onViewChange('add')} />
            <Flex direction={'column'} gap={'size-100'}>
                {sources.map((source) => {
                    const isActiveSource = activeSource?.id === source.id;

                    if (isWebcamSource(source)) {
                        return (
                            <WebcamSourceReadOnly
                                key={source.id}
                                isActive={isActiveSource}
                                source={source}
                                menuItems={getMenuItems({ isActiveSource, isActiveProject })}
                                onAction={handleAction(source)}
                            />
                        );
                    }

                    if (isImagesFolderSource(source)) {
                        return (
                            <ImagesFolderSourceReadOnly
                                key={source.id}
                                isActive={isActiveSource}
                                source={source}
                                menuItems={getMenuItems({ isActiveSource, isActiveProject })}
                                onAction={handleAction(source)}
                            />
                        );
                    }
                })}
            </Flex>
        </Flex>
    );
};
