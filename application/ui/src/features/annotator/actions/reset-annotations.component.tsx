/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Tooltip, TooltipTrigger } from '@geti/ui';
import { Refresh } from '@geti/ui/icons';

import { useUndoRedo } from './undo-redo/undo-redo-provider.component';

export const ResetAnnotations = () => {
    const { reset } = useUndoRedo();

    const resetAnnotations = () => {
        reset([]);
    };

    return (
        <TooltipTrigger>
            <ActionButton isQuiet aria-label={'Reset annotations'} onPress={resetAnnotations}>
                <Refresh />
            </ActionButton>
            <Tooltip>Reset annotations</Tooltip>
        </TooltipTrigger>
    );
};
