/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagesFolderSourceType } from '@/api';

export const getMockedImagesFolderSource = (
    source: Partial<{ active: boolean; imagesFolderPath: string }> = {}
): ImagesFolderSourceType => {
    return {
        id: '123',
        active: source.active ?? true,
        config: {
            seekable: true,
            images_folder_path: source.imagesFolderPath ?? '',
            source_type: 'images_folder',
        },
    };
};
