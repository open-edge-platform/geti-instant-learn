/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { vi } from 'vitest';

/* eslint-disable no-underscore-dangle */
export const setMockedTauriContext = () => {
    window.__TAURI__ = {
        core: {
            invoke: vi.fn(),
        },
    };
};

export const clearMockedTauriContext = () => {
    window.__TAURI__ = undefined;
};
/* eslint-enable no-underscore-dangle */
