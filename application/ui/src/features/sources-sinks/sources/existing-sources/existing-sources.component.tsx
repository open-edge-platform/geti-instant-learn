/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Source } from '@geti-prompt/api';
import { useCurrentProject } from '@geti-prompt/hooks';
import { Button, Flex } from '@geti/ui';
import { Add as AddIcon } from '@geti/ui/icons';
import { orderBy } from 'lodash-es';

import { useDeleteSource } from '../api/use-delete-source';
import { useUpdateSource } from '../api/use-update-source';
import { ImagesFolderSourceCard } from '../images-folder/images-folder-card.component';
import { TestDatasetCard } from '../test-dataset/test-dataset-card.component';
import { isImagesFolderSource, isTestDatasetSource, isWebcamSource, SourcesViews } from '../utils';
import { WebcamSourceCard } from '../webcam/webcam-source-card.component';

import styles from './existing-sources.module.scss';

const getMenuItems = ({
    isActiveProject,
    isActiveSource,
    isTestDataset,
}: {
    isActiveProject: boolean;
    isActiveSource: boolean;
    isTestDataset: boolean;
}) => {
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
        if (item.key === 'edit' && isTestDataset) {
            return false;
        }
        return true;
    });
};

const sortSources = (sources: Source[]): Source[] => {
    return orderBy(sources, (source) => source.connected, 'desc');
};

interface ExistingSourcesListProps {
    sources: Source[];
    onViewChange: (view: SourcesViews) => void;
    onSetSourceInEditionId: (sourceId: string) => void;
}

const ExistingSourcesList = ({ sources, onSetSourceInEditionId, onViewChange }: ExistingSourcesListProps) => {
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
            deleteSource.mutate(
                { params: { path: { project_id: project.id, source_id: source.id } } },
                {
                    onSuccess: () => {
                        if (sources.length === 1) {
                            onViewChange('list');
                        }
                    },
                }
            );
        }
    };

    return (
        <Flex direction={'column'} gap={'size-100'}>
            {sortSources(sources).map((source) => {
                const isActiveSource = source.connected;

                if (isTestDatasetSource(source)) {
                    return (
                        <TestDatasetCard
                            key={source.id}
                            source={source}
                            menuItems={getMenuItems({ isActiveSource, isActiveProject, isTestDataset: true })}
                            onAction={handleAction(source)}
                        />
                    );
                }

                if (isWebcamSource(source)) {
                    return (
                        <WebcamSourceCard
                            key={source.id}
                            source={source}
                            menuItems={getMenuItems({ isActiveSource, isActiveProject, isTestDataset: false })}
                            onAction={handleAction(source)}
                        />
                    );
                }

                if (isImagesFolderSource(source)) {
                    return (
                        <ImagesFolderSourceCard
                            key={source.id}
                            source={source}
                            menuItems={getMenuItems({ isActiveSource, isActiveProject, isTestDataset: false })}
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
    onViewChange: (view: SourcesViews) => void;
    onSetSourceInEditionId: (sourceId: string) => void;
}

export const ExistingSources = ({ sources, onViewChange, onSetSourceInEditionId }: ExistingSourcesProps) => {
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
                onViewChange={onViewChange}
                onSetSourceInEditionId={onSetSourceInEditionId}
            />
        </Flex>
    );
};
