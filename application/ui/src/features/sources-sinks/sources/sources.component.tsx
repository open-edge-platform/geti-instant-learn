/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, useState } from 'react';

import { SourceType } from '@geti-prompt/api';
import { useGetSources } from '@geti-prompt/hooks';
import { ImagesFolder as ImagesFolderIcon, WebCam } from '@geti-prompt/icons';
import { ActionButton, Divider, Flex, Heading, View } from '@geti/ui';
import { Back, Datasets } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import TestDatasetImg from '../../../../../backend/backend/.data/templates/datasets/coffee-berries/scene00001.png';
import { DisclosureGroup } from '../disclosure-group/disclosure-group.component';
import { EditSource } from './edit-sources/edit-sources.component';
import { ExistingSources } from './existing-sources/existing-sources.component';
import { ImagesFolder } from './images-folder/images-folder.component';
import { TestDataset } from './test-dataset/test-dataset.component';
import { SourcesViews } from './utils';
import { WebcamSource } from './webcam/webcam-source.component';

interface SourcesList {
    onViewChange: (view: SourcesViews) => void;
}

const SourcesList = ({ onViewChange }: SourcesList) => {
    const sourcesList: {
        label: string;
        value: SourceType;
        content: ReactNode;
        icon: ReactNode;
    }[] = [
        {
            label: 'Webcam',
            value: 'webcam',
            content: <WebcamSource onSaved={() => onViewChange('existing')} />,
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
            content: <ImagesFolder onSaved={() => onViewChange('existing')} />,
            icon: <ImagesFolderIcon width={'24px'} />,
        },
        {
            label: 'Test dataset',
            value: 'images_folder',
            content: (
                <TestDataset
                    title={'Coffee Bean Quality Dataset'}
                    description={
                        'A ready-to-use collection of coffee bean images—ripe and unripe—for instant inference and ' +
                        'testing. Select it to experience the pipeline in action right away.'
                    }
                    // TODO: update the path once backend supports it
                    folderPath={'backend/backend/.data/templates/datasets/' + 'coffee-berries'}
                    imgSrc={TestDatasetImg}
                    onSaved={() => onViewChange('existing')}
                />
            ),
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
        <View>
            <Flex alignItems={'center'} gap={'size-75'}>
                <ActionButton isQuiet onPress={() => onViewChange('existing')}>
                    <Back />
                </ActionButton>
                <Heading margin={0}>Add new input source</Heading>
            </Flex>
            <Divider size={'S'} marginY={'size-200'} />

            <SourcesList onViewChange={onViewChange} />
        </View>
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
