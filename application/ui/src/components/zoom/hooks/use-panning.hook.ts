/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { useEventListener } from '@geti-prompt/hooks';

export const usePanning = () => {
    const [isPanning, setIsPanning] = useState(false);

    useEventListener('keydown', (event: KeyboardEvent) => {
        if (event.code === 'Space') {
            event.preventDefault();
            setIsPanning(true);
        }
    });

    useEventListener('keyup', (event: KeyboardEvent) => {
        if (event.code === 'Space') {
            setIsPanning(false);
        }
    });

    return { isPanning, setIsPanning };
};
