/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { Source } from '@geti-prompt/api';

import { PipelineEntityPanel } from '../../pipeline-entity-panel/pipeline-entity-panel.component';
import { EditImagesFolder } from '../images-folder/edit-images-folder.component';
import { EditUsbCameraSource } from '../usb-camera/edit-usb-camera-source.component';
import { isImagesFolderSource, isUsbCameraSource, isVideoFileSource, SourcesViews } from '../utils';
import { EditVideoFile } from '../video-file/edit-video-file.component';

interface EditSourceContainerProps {
    children: ReactNode;
    onBackClick: () => void;
    title: string;
}

const EditSourceContainer = ({ children, onBackClick, title }: EditSourceContainerProps) => {
    return (
        <PipelineEntityPanel
            title={<PipelineEntityPanel.Title>Edit input source</PipelineEntityPanel.Title>}
            onBackClick={onBackClick}
        >
            <PipelineEntityPanel.Content title={title}>{children}</PipelineEntityPanel.Content>
        </PipelineEntityPanel>
    );
};

interface EditSourceProps {
    source: Source;
    onViewChange: (view: SourcesViews) => void;
}

export const EditSource = ({ source, onViewChange }: EditSourceProps) => {
    const handleGoBack = () => onViewChange('existing');

    if (isUsbCameraSource(source)) {
        return (
            <EditSourceContainer onBackClick={handleGoBack} title={'USB Camera'}>
                <EditUsbCameraSource source={source} onSaved={handleGoBack} />
            </EditSourceContainer>
        );
    }

    if (isImagesFolderSource(source)) {
        return (
            <EditSourceContainer onBackClick={handleGoBack} title={'Images folder'}>
                <EditImagesFolder source={source} onSaved={handleGoBack} />
            </EditSourceContainer>
        );
    }

    if (isVideoFileSource(source)) {
        return (
            <EditSourceContainer onBackClick={handleGoBack} title={'Video file'}>
                <EditVideoFile source={source} onSaved={handleGoBack} />
            </EditSourceContainer>
        );
    }

    throw new Error(`Source type "${source.config.source_type}" is not supported for editing.`);
};
