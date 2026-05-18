/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback } from 'react';

import { useUpdateProject } from '../../features/project/api/use-update-project.hook';
import { useCurrentProject } from './use-current-project.hook';

export type PromptMode = 'TEXT' | 'VISUAL';

const toBackendMode = (mode: string): PromptMode => {
    if (mode.toLowerCase().includes('text')) {
        return 'TEXT';
    }
    return 'VISUAL';
};

export const usePromptMode = (): [PromptMode, (mode: string) => void] => {
    const { data: project } = useCurrentProject();
    const updateProject = useUpdateProject();

    const handleModeChange = useCallback(
        (option: string) => {
            const backendMode = toBackendMode(option);

            if (backendMode === project.prompt_mode) return;

            updateProject.mutate(project.id, { prompt_mode: backendMode });
        },
        [project.id, project.prompt_mode, updateProject]
    );

    return [project.prompt_mode as PromptMode, handleModeChange] as const;
};
