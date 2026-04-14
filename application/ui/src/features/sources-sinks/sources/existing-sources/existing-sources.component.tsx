/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef } from 'react';

import { Source, SourceType } from '@/api';
import { useCurrentProject } from '@/hooks';
import { toast } from '@geti/ui';
import { orderBy } from 'lodash-es';

import { ExistingPipelineEntities } from '../../existing-pipeline-entities/existing-pipeline-entities.component';
import { useDeleteSource } from '../api/use-delete-source';
import { useUpdateSource } from '../api/use-update-source';
import { ImagesFolderSourceCard } from '../images-folder/images-folder-card.component';
import { SampleDatasetCard } from '../sample-dataset/sample-dataset-card.component';
import { UsbCameraSourceCard } from '../usb-camera/usb-camera-source-card.component';
import {
    isImagesFolderSource,
    isTestDatasetSource,
    isUsbCameraSource,
    isVideoFileSource,
    SourcesViews,
} from '../utils';
import { VideoFileCard } from '../video-file/video-file-card.component';

const getMenuItems = ({
    isActiveSource,
    isTestDataset,
    isAvailable,
}: {
    isActiveSource: boolean;
    isTestDataset: boolean;
    isAvailable: boolean;
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
        if (item.key === 'connect' && (isActiveSource || !isAvailable)) {
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
    const notifiedSourceIds = useRef(new Set<string>());

    const updateSource = useUpdateSource();
    const deleteSource = useDeleteSource();

    // Show toast notifications for unavailable sources
    useEffect(() => {
        sources.forEach((source) => {
            // Only show notification if source is unavailable and we haven't notified about it yet
            if (source.available === false && source.unavailable_reason && !notifiedSourceIds.current.has(source.id)) {
                const sourceTypeLabel =
                    source.config.source_type === 'video_file'
                        ? 'Video file'
                        : source.config.source_type === 'images_folder'
                          ? 'Images folder'
                          : 'Source';

                toast({
                    type: 'error',
                    message: `${sourceTypeLabel} unavailable: ${source.unavailable_reason}`,
                    duration: 8000,
                });

                notifiedSourceIds.current.add(source.id);
            }

            // Remove from notified set if source becomes available again
            if (source.available !== false && notifiedSourceIds.current.has(source.id)) {
                notifiedSourceIds.current.delete(source.id);
            }
        });
    }, [sources]);

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
                const isAvailable = source.available !== false;

                const menuItems = getMenuItems({
                    isActiveSource,
                    isTestDataset: isTestDatasetSource(source),
                    isAvailable,
                });

                if (isTestDatasetSource(source)) {
                    return (
                        <SampleDatasetCard
                            key={source.id}
                            source={source}
                            menuItems={menuItems}
                            onAction={handleAction(source)}
                        />
                    );
                }

                if (isUsbCameraSource(source)) {
                    return (
                        <UsbCameraSourceCard
                            key={source.id}
                            source={source}
                            menuItems={menuItems}
                            onAction={handleAction(source)}
                        />
                    );
                }

                if (isImagesFolderSource(source)) {
                    return (
                        <ImagesFolderSourceCard
                            key={source.id}
                            source={source}
                            menuItems={menuItems}
                            onAction={handleAction(source)}
                        />
                    );
                }

                if (isVideoFileSource(source)) {
                    return (
                        <VideoFileCard
                            key={source.id}
                            source={source}
                            menuItems={menuItems}
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

const AVAILABLE_SOURCE_TYPES: SourceType[] = ['usb_camera', 'images_folder', 'sample_dataset', 'video_file'];

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
