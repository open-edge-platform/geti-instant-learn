/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Source, SourceType } from '@geti-prompt/api';
import { useCurrentProject } from '@geti-prompt/hooks';
import { orderBy } from 'lodash-es';

import { ExistingPipelineEntities } from '../../existing-pipeline-entities/existing-pipeline-entities.component';
import { useDeleteSource } from '../api/use-delete-source';
import { useUpdateSource } from '../api/use-update-source';
import { ImagesFolderSourceCard } from '../images-folder/images-folder-card.component';
import { SampleDatasetCard } from '../sample-dataset/sample-dataset-card.component';
import { isImagesFolderSource, isTestDatasetSource, isWebcamSource, SourcesViews } from '../utils';
import { WebcamSourceCard } from '../webcam/webcam-source-card.component';

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
    return orderBy(sources, (source) => source.active, 'desc');
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
                active: true,
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
        <ExistingPipelineEntities.List>
            {sortSources(sources).map((source) => {
                const isActiveSource = source.active;

                if (isTestDatasetSource(source)) {
                    return (
                        <SampleDatasetCard
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
        </ExistingPipelineEntities.List>
    );
};

interface ExistingSourcesProps {
    sources: Source[];
    onViewChange: (view: SourcesViews) => void;
    onSetSourceInEditionId: (sourceId: string) => void;
}

const AVAILABLE_SOURCE_TYPES: SourceType[] = ['webcam', 'images_folder', 'sample_dataset'];

export const ExistingSources = ({ sources, onViewChange, onSetSourceInEditionId }: ExistingSourcesProps) => {
    const canCreateSource = !AVAILABLE_SOURCE_TYPES.every((type) =>
        sources.some((source) => source.config.source_type === type)
    );

    return (
        <ExistingPipelineEntities
            addNewEntityButton={
                canCreateSource && (
                    <ExistingPipelineEntities.AddNewEntityButton
                        text={'Add new source'}
                        onPress={() => onViewChange('add')}
                    />
                )
            }
        >
            <ExistingSourcesList
                sources={sources}
                onViewChange={onViewChange}
                onSetSourceInEditionId={onSetSourceInEditionId}
            />
        </ExistingPipelineEntities>
    );
};
