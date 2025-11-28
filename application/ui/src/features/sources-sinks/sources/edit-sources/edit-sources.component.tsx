/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { Source } from '@geti-prompt/api';
import { ActionButton, Divider, Flex, Heading, View } from '@geti/ui';
import { Back } from '@geti/ui/icons';

import { EditImagesFolder } from '../images-folder/edit-images-folder.component';
import { isImagesFolderSource, isWebcamSource, SourcesViews } from '../utils';
import { EditWebcamSource } from '../webcam/edit-webcam-source.component';

interface EditSourceContainerProps {
    children: ReactNode;
    onBackClick: () => void;
    title: string;
}

const EditSourceContainer = ({ children, onBackClick, title }: EditSourceContainerProps) => {
    return (
        <View>
            <Flex alignItems={'center'} gap={'size-75'}>
                <ActionButton isQuiet onPress={onBackClick}>
                    <Back />
                </ActionButton>
                <Heading margin={0}>Edit input source</Heading>
            </Flex>
            <Divider size={'S'} marginY={'size-200'} />
            <View
                padding={'size-200'}
                backgroundColor={'gray-100'}
                borderColor={'gray-200'}
                borderWidth={'thin'}
                borderRadius={'regular'}
            >
                <Heading margin={0} marginBottom={'size-200'}>
                    {title}
                </Heading>
                {children}
            </View>
        </View>
    );
};

interface EditSourceProps {
    source: Source;
    onViewChange: (view: SourcesViews) => void;
}

export const EditSource = ({ source, onViewChange }: EditSourceProps) => {
    const handleGoBack = () => onViewChange('existing');

    if (isWebcamSource(source)) {
        return (
            <EditSourceContainer onBackClick={handleGoBack} title={'Webcam'}>
                <EditWebcamSource source={source} onSaved={handleGoBack} />
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

    throw new Error(`Source type "${source.config.source_type}" is not supported for editing.`);
};
