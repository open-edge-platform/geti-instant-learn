/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { ActionButton } from '@geti/ui';
import { Invisible, Visible } from '@geti/ui/icons';

export const ToggleAnnotationsVisibility = () => {
    const [isVisible, setIsVisible] = useState<boolean>(true);

    return (
        <ActionButton
            aria-label={isVisible ? 'Hide annotations' : 'Show annotations'}
            onPress={() => setIsVisible((prev) => !prev)}
            isQuiet
        >
            {isVisible ? <Visible /> : <Invisible />}
        </ActionButton>
    );
};
