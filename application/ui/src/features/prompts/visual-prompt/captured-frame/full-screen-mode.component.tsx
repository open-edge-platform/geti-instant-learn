/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton } from '@geti/ui';
import { Collapse, Expand } from '@geti/ui/icons';

interface FullScreenModeProps {
    isFullScreenMode: boolean;
    onFullScreenModeChange: (isFullScreenMode: boolean) => void;
}

export const FullScreenMode = ({ isFullScreenMode, onFullScreenModeChange }: FullScreenModeProps) => {
    return (
        <ActionButton
            isQuiet
            aria-label={isFullScreenMode ? 'Close full screen' : 'Open full screen'}
            onPress={() => onFullScreenModeChange(!isFullScreenMode)}
        >
            {isFullScreenMode ? <Collapse /> : <Expand />}
        </ActionButton>
    );
};
