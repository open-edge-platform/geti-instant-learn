/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Source } from '@/api';

export const getMockedSource = (source?: Partial<Source>): Source => {
    return {
        id: 'source-1',
        active: false,
        config: {
            source_type: 'usb_camera',
            device_id: 0,
            seekable: false,
        },
        ...source,
    };
};
