/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, useState } from 'react';

import { Source, SourceType } from '@geti-prompt/api';
import { useGetSources } from '@geti-prompt/hooks';
import { ImagesFolder as ImagesFolderIcon, WebCam } from '@geti-prompt/icons';
import { ActionButton, Divider, Flex, Heading, View } from '@geti/ui';
import { Back } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import { DisclosureGroup } from '../disclosure-group/disclosure-group.component';
import { EditSource } from './edit-sources-list.component';
import { ExistingSourcesList } from './existing-sources-list.component';
import { ImagesFolder } from './images-folder/images-folder.component';
import { getImagesFolderSource, getWebcamSource, SourcesViews } from './utils';
import { WebcamSource } from './webcam/webcam-source.component';

interface SourcesList {
    sources: Source[] | undefined;
    activeSource: Source | undefined;
    onViewChange: (view: SourcesViews) => void;
}

const SourcesList = ({ sources, activeSource, onViewChange }: SourcesList) => {
    const sourcesList: {
        label: string;
        value: SourceType;
        content: ReactNode;
        icon: ReactNode;
    }[] = [
        {
            label: 'Webcam',
            value: 'webcam',
            content: <WebcamSource source={getWebcamSource(sources)} onSaved={() => onViewChange('existing')} />,
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
            content: <ImagesFolder source={getImagesFolderSource(sources)} onSaved={() => onViewChange('existing')} />,
            icon: <ImagesFolderIcon width={'24px'} />,
        },
    ];

    return <DisclosureGroup items={sourcesList} value={activeSource?.config.source_type} />;
};

interface AddSourceProps {
    onViewChange: (view: SourcesViews) => void;
}

const AddSource = ({ onViewChange }: AddSourceProps) => {
    return (
        <View>
            <ActionButton isQuiet onPress={() => onViewChange('existing')} width={'100%'}>
                <Flex alignItems={'center'} gap={'size-100'} width={'100%'} justifyContent={'start'}>
                    <Back /> <Heading margin={0}>Add new input source</Heading>
                </Flex>
            </ActionButton>
            <Divider size={'S'} marginY={'size-200'} />

            <SourcesList sources={[]} activeSource={undefined} onViewChange={onViewChange} />
        </View>
    );
};

export const Sources = () => {
    const { data } = useGetSources();
    const activeSource = data.sources.find((source) => source.connected);
    const [view, setView] = useState<SourcesViews>(isEmpty(data.sources) ? 'list' : 'existing');
    const [sourceInEditionId, setSourceInEditionId] = useState<string | null>(null);
    const sourceInEdition = data.sources.find((source) => source.id === sourceInEditionId);

    if (view === 'existing') {
        return (
            <ExistingSourcesList
                sources={data.sources}
                activeSource={activeSource}
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

    return <SourcesList sources={data.sources} activeSource={activeSource} onViewChange={setView} />;
};
