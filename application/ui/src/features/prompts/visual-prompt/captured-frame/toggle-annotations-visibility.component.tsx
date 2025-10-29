/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton } from '@geti/ui';
import { Invisible, Visible } from '@geti/ui/icons';

import { useAnnotationVisibility } from '../../../annotator/providers/annotation-visibility-provider.component';

export const ToggleAnnotationsVisibility = () => {
    const { isVisible, toggleVisibility } = useAnnotationVisibility();

    return (
        <ActionButton aria-label={`${isVisible ? 'Hide' : 'Show'} annotations`} isQuiet onPress={toggleVisibility}>
            {isVisible ? <Visible /> : <Invisible />}
        </ActionButton>
    );
};
