/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Grid, minmax, repeat } from '@geti/ui';

import TestImage from '../../../../../assets/test.webp';
import { PromptThumbnail } from '../prompt-thumbnail/prompt-thumbnail.component';

const PROMPT_THUMBNAILS = [
    { url: TestImage, frameId: 'frame-1' },
    { url: TestImage, frameId: 'frame-2' },
    { url: TestImage, frameId: 'frame-3' },
    { url: TestImage, frameId: 'frame-4' },
    { url: TestImage, frameId: 'frame-5' },
    { url: TestImage, frameId: 'frame-6' },
];

const usePrompts = () => {
    // TODO: GET `/api/v1/projects/{project_id}/prompts`
    // once the backend is implemented.
    return PROMPT_THUMBNAILS;
};

export const PromptThumbnailList = () => {
    const promptThumbnails = usePrompts();

    if (promptThumbnails.length === 0) {
        return null;
    }

    return (
        <Grid columns={[repeat('auto-fit', minmax('size-1600', '1fr'))]} gap={'size-100'}>
            {promptThumbnails.map((image) => (
                <PromptThumbnail key={image.frameId} image={image} />
            ))}
        </Grid>
    );
};
