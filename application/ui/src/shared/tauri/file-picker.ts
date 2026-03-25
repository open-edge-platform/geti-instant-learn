/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable no-underscore-dangle */
const isTauriContext = (): boolean => typeof window.__TAURI__?.core?.invoke === 'function';
/* eslint-enable no-underscore-dangle */

const getSingleSelectedPath = (selectedPath: string | string[] | null): string | null => {
    if (typeof selectedPath === 'string') {
        return selectedPath;
    }

    if (Array.isArray(selectedPath)) {
        return selectedPath.length > 0 ? selectedPath[0] : null;
    }
    return null;
};

export const pickVideoFilePath = async (): Promise<string | null> => {
    if (!isTauriContext()) {
        return null;
    }

    const { open } = await import('@tauri-apps/plugin-dialog');
    const selectedPath = await open({
        directory: false,
        multiple: false,
        filters: [
            {
                name: 'Videos',
                extensions: ['mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v'],
            },
        ],
    });

    return getSingleSelectedPath(selectedPath);
};

export const pickFolderPath = async (): Promise<string | null> => {
    if (!isTauriContext()) {
        return null;
    }

    const { open } = await import('@tauri-apps/plugin-dialog');
    const selectedPath = await open({
        directory: true,
        multiple: false,
    });

    return getSingleSelectedPath(selectedPath);
};

export const isTauriRuntime = isTauriContext;
