/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { dimensionValue, Grid, minmax } from '@geti/ui';

import { usePromptMode } from '../prompts/prompt-modes/prompt-modes.component';
import { CaptureFrameButton } from './capture-frame-button.component';
import { ImagesFolderStream } from './images-folder-stream/images-folder-stream.component';
import { Video } from './video.component';

const useActiveSource = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useQuery('get', '/api/v1/projects/{project_id}/sources', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    return data?.sources.find((source) => source.connected);
};

const WebcamStream = () => {
    const promptMode = usePromptMode();

    return (
        <Grid
            height={'100%'}
            width={'100%'}
            rows={[minmax(0, '1fr'), 'max-content']}
            rowGap={'size-200'}
            UNSAFE_style={{ paddingTop: dimensionValue('size-600'), paddingBottom: dimensionValue('size-200') }}
        >
            <Video />
            {promptMode === 'visual' && <CaptureFrameButton />}
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
