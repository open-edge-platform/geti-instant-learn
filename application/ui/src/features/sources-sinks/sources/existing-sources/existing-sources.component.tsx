/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Source } from '@geti-prompt/api';
import { useCurrentProject } from '@geti-prompt/hooks';
import { Button, Flex } from '@geti/ui';
import { Add as AddIcon } from '@geti/ui/icons';

import { useDeleteSource } from '../api/use-delete-source';
import { useUpdateSource } from '../api/use-update-source';
import { ImagesFolderSourceCard } from '../images-folder/images-folder-card.component';
import { isImagesFolderSource, isWebcamSource, SourcesViews } from '../utils';
import { WebcamSourceCard } from '../webcam/webcam-source-card.component';

import styles from './existing-sources.module.scss';

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

interface ExistingSourcesListProps {
    sources: Source[];
    activeSource: Source | undefined;
    onViewChange: (view: SourcesViews) => void;
    onSetSourceInEditionId: (sourceId: string) => void;
}

const ExistingSourcesList = ({
    sources,
    activeSource,
    onSetSourceInEditionId,
    onViewChange,
}: ExistingSourcesListProps) => {
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
        <Flex direction={'column'} gap={'size-100'}>
            {sources.map((source) => {
                const isActiveSource = activeSource?.id === source.id;

                if (isWebcamSource(source)) {
                    return (
                        <WebcamSourceCard
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
                        <ImagesFolderSourceCard
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
    );
};

interface ExistingSourcesProps {
    sources: Source[];
    activeSource: Source | undefined;
    onViewChange: (view: SourcesViews) => void;
    onSetSourceInEditionId: (sourceId: string) => void;
}

export const ExistingSources = ({
    sources,
    onViewChange,
    activeSource,
    onSetSourceInEditionId,
}: ExistingSourcesProps) => {
    return (
        <Flex direction={'column'} gap={'size-200'}>
            <Button
                variant={'secondary'}
                onPress={() => onViewChange('add')}
                UNSAFE_className={styles.addNewSourceButton}
            >
                <AddIcon /> Add new source
            </Button>
            <ExistingSourcesList
                sources={sources}
                activeSource={activeSource}
                onViewChange={onViewChange}
                onSetSourceInEditionId={onSetSourceInEditionId}
            />
        </Flex>
    );
};
