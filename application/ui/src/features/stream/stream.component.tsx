/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useGetSources, usePromptMode } from '@geti-prompt/hooks';
import { dimensionValue, Grid, minmax, View } from '@geti/ui';

import { CaptureFrameButton } from './capture-frame-button.component';
import { ImagesFolderStream } from './images-folder-stream/images-folder-stream.component';
import { Video } from './video.component';

const useActiveSource = () => {
    const { data } = useGetSources();

    return data.sources.find((source) => source.connected);
};

const WebcamStream = () => {
    const promptMode = usePromptMode();

    return (
        <Grid
            height={'100%'}
            width={'100%'}
            rows={[minmax(0, '1fr'), 'max-content']}
            columns={['size-200', minmax(0, '1fr'), 'size-200']}
            areas={['left-gutter video right-gutter', 'left-gutter capture right-gutter']}
            rowGap={'size-200'}
            UNSAFE_style={{ paddingTop: dimensionValue('size-600'), paddingBottom: dimensionValue('size-200') }}
        >
            <View gridArea={'video'}>
                <Video />
            </View>
            <View gridArea={'capture'} justifySelf={'center'}>
                {promptMode === 'visual' && <CaptureFrameButton />}
            </View>
        </Grid>
    );
};

export const Stream = () => {
    const activeSource = useActiveSource();

    // Should never happen, just for type safety
    if (activeSource === undefined) {
        return null;
    }

    if (activeSource.config.source_type === 'webcam') {
        return <WebcamStream />;
    }

    if (activeSource.config.source_type === 'images_folder') {
        return <ImagesFolderStream />;
    }
};
