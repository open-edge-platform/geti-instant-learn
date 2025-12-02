/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, useState } from 'react';

import { SourceType } from '@geti-prompt/api';
import { useGetSources } from '@geti-prompt/hooks';
import { ImagesFolder as ImagesFolderIcon, WebCam } from '@geti-prompt/icons';
import { Datasets } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import { DisclosureGroup } from '../disclosure-group/disclosure-group.component';
import { PipelineEntityPanel } from '../pipeline-entity-panel/pipeline-entity-panel.component';
import { EditSource } from './edit-sources/edit-sources.component';
import { ExistingSources } from './existing-sources/existing-sources.component';
import { CreateImagesFolder } from './images-folder/create-images-folder.component';
import { CreateSampleDataset } from './sample-dataset/create-sample-dataset.component';
import { SourcesViews } from './utils';
import { CreateWebcamSource } from './webcam/create-webcam-source.component';

interface SourcesList {
    onViewChange: (view: SourcesViews) => void;
}

const SourcesList = ({ onViewChange }: SourcesList) => {
    const navigateToExistingView = () => {
        onViewChange('existing');
    };

    const sourcesList: {
        label: string;
        value: SourceType;
        content: ReactNode;
        icon: ReactNode;
    }[] = [
        {
            label: 'Webcam',
            value: 'webcam',
            content: <CreateWebcamSource onSaved={navigateToExistingView} />,
            icon: <WebCam width={'24px'} />,
        },
        /*{
            label: 'IP Camera',
            value: 'ip_camera',
            content: <IPCameraForm />,
            icon: <IPCamera width={'24px'} />,
        },*/
        /*{ label: 'GenICam', value: 'gen-i-cam', content: 'Test', icon: <GenICam width={'24px'} /> },*/
        /*{
            label: 'Video file',
            value: 'video_file',
            content: 'Test',
            icon: <VideoFile width={'24px'} />,
        },*/
        {
            label: 'Image folder',
            value: 'images_folder',
            content: <CreateImagesFolder onSaved={navigateToExistingView} />,
            icon: <ImagesFolderIcon width={'24px'} />,
        },
        {
            label: 'Sample dataset',
            value: 'sample_dataset',
            content: <CreateSampleDataset onSaved={navigateToExistingView} />,
            icon: <Datasets width={'24px'} />,
        },
    ];

    return <DisclosureGroup items={sourcesList} />;
};

interface AddSourceProps {
    onViewChange: (view: SourcesViews) => void;
}

const AddSource = ({ onViewChange }: AddSourceProps) => {
    return (
        <PipelineEntityPanel
            title={<PipelineEntityPanel.Title>Add new input source</PipelineEntityPanel.Title>}
            onBackClick={() => onViewChange('existing')}
        >
            <SourcesList onViewChange={onViewChange} />
        </PipelineEntityPanel>
    );
};

export const Sources = () => {
    const { data } = useGetSources();
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
        return <AddSource onViewChange={setView} />;
    }

    return <SourcesList onViewChange={setView} />;
};
