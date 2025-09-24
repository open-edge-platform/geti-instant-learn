/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton } from '@geti/ui';
import { Adjustments } from '@geti/ui/icons';

export const CanvasAdjustments = () => {
    return (
        <ActionButton aria-label={'Canvas adjustments'} isQuiet>
            <Adjustments />
        </ActionButton>
    );
};
