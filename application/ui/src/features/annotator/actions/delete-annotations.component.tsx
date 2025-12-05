/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Tooltip, TooltipTrigger } from '@geti/ui';
import { Refresh } from '@geti/ui/icons';

import { useAnnotationActions } from '../providers/annotation-actions-provider.component';

export const DeleteAnnotations = () => {
    const { deleteAllAnnotations } = useAnnotationActions();

    return (
        <TooltipTrigger>
            <ActionButton isQuiet aria-label={'Delete annotations'} onPress={deleteAllAnnotations}>
                <Refresh />
            </ActionButton>
            <Tooltip>Delete annotations</Tooltip>
        </TooltipTrigger>
    );
};
