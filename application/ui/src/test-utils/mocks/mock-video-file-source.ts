/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { VideoFileSourceType } from '@geti-prompt/api';

export const getMockedVideoFileSource = (
    source: Partial<{ active: boolean; filePath: string; id: string }> = {}
): VideoFileSourceType => {
    return {
        id: source.id ?? '123',
        active: source.active ?? true,
        config: {
            seekable: true,
            video_path: source.filePath ?? '',
            source_type: 'video_file',
        },
    };
};
