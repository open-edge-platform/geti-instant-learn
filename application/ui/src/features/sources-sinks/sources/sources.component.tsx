/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, Suspense, useState } from 'react';

import { Source, SourceType } from '@/api';
import { useGetSources } from '@/hooks';
import { ImagesFolder as ImagesFolderIcon, UsbCamera, VideoFile } from '@/icons';
import { Loading } from '@geti/ui';
import { Datasets } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import { DisclosureGroup } from '../disclosure-group/disclosure-group.component';
import { PipelineEntityPanel } from '../pipeline-entity-panel/pipeline-entity-panel.component';
import { usePrefetchAvailableSources } from './api/use-available-sources';
import { EditSource } from './edit-sources/edit-sources.component';
import { ExistingSources } from './existing-sources/existing-sources.component';
import { CreateImagesFolder } from './images-folder/create-images-folder.component';
import { CreateSampleDataset } from './sample-dataset/create-sample-dataset.component';
import { CreateUsbCameraSource } from './usb-camera/create-usb-camera-source.component';
import { SourcesViews } from './utils';
import { CreateVideoFile } from './video-file/create-video-file.component';

interface SourcesList {
    onViewChange: (view: SourcesViews) => void;
    sources: Source[];
}

const SourcesList = ({ onViewChange, sources }: SourcesList) => {
    const navigateToExistingView = () => {
        onViewChange('existing');
    };

    const sourcesList = [
        {
            label: 'USB Camera',
            value: 'usb_camera',
            content: (
                <Suspense fallback={<Loading mode={'inline'} size={'S'} />}>
                    <CreateUsbCameraSource onSaved={navigateToExistingView} />
                </Suspense>
            ),
            icon: <UsbCamera width={'24px'} />,
        },
        /*{
            label: 'IP Camera',
            value: 'ip_camera',
            content: <IPCameraForm />,
            icon: <IPCamera width={'24px'} />,
        },*/
        /*{ label: 'GenICam', value: 'gen-i-cam', content: 'Test', icon: <GenICam width={'24px'} /> },*/

        {
            label: 'Image folder',
            value: 'images_folder',
            content: <CreateImagesFolder onSaved={navigateToExistingView} />,
            icon: <ImagesFolderIcon width={'24px'} />,
        },
        {
            label: 'Video file',
            value: 'video_file',
            content: <CreateVideoFile onSaved={navigateToExistingView} />,
            icon: <VideoFile width={'24px'} />,
        },
        {
            label: 'Sample dataset',
            value: 'sample_dataset',
            content: <CreateSampleDataset onSaved={navigateToExistingView} />,
            icon: <Datasets width={'24px'} />,
        },
    ] satisfies { label: string; value: SourceType; content: ReactNode; icon: ReactNode }[];

    const visibleSourcesList = sourcesList.filter(
        (source) => !sources.some((existingSource) => existingSource.config.source_type === source.value)
    );

    const defaultOpenSource = visibleSourcesList.length === 1 ? visibleSourcesList[0].value : undefined;

    return <DisclosureGroup items={visibleSourcesList} value={defaultOpenSource} />;
};

interface AddSourceProps {
    sources: Source[];
    onViewChange: (view: SourcesViews) => void;
}

const AddSource = ({ onViewChange, sources }: AddSourceProps) => {
    return (
        <PipelineEntityPanel
            title={<PipelineEntityPanel.Title>Add new input source</PipelineEntityPanel.Title>}
            onBackClick={() => onViewChange('existing')}
        >
            <SourcesList onViewChange={onViewChange} sources={sources} />
        </PipelineEntityPanel>
    );
};

export const Sources = () => {
    const { data } = useGetSources();
    usePrefetchAvailableSources('usb_camera');

    const [view, setView] = useState<SourcesViews>(isEmpty(data.sources) ? 'list' : 'existing');
    const [sourceInEditionId, setSourceInEditionId] = useState<string | null>(null);
    const sourceInEdition = data.sources.find((source) => source.id === sourceInEditionId);

    if (view === 'existing') {
        return (
            <ExistingSources
                sources={data.sources}
                onViewChange={setView}
                onSetSourceInEditionId={setSourceInEditionId}
            />
        );
    }

    if (view === 'edit' && sourceInEdition !== undefined) {
        return <EditSource source={sourceInEdition} onViewChange={setView} />;
    }

    if (view === 'add') {
        return <AddSource onViewChange={setView} sources={data.sources} />;
    }

    return <SourcesList onViewChange={setView} sources={data.sources} />;
};
