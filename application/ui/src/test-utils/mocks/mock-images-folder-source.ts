/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagesFolderSourceType } from '@geti-prompt/api';

export const getMockedImagesFolderSource = (
    source: Partial<{ connected: boolean; imagesFolderPath: string }> = {}
): ImagesFolderSourceType => {
    return {
        id: '123',
        connected: source.connected ?? true,
        config: {
            seekable: true,
            images_folder_path: source.imagesFolderPath ?? '',
            source_type: 'images_folder',
        },
    };
};
