/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Invisible, Visible } from '@geti/ui/icons';
import { IconWrapper } from 'src/components/icon-wrapper/icon-wrapper.component';
import { useAnnotationVisibility } from 'src/features/annotator/providers/annotation-visibility-provider.component';

export const ToggleAnnotationsVisibility = () => {
    const { isVisible, toggleVisibility } = useAnnotationVisibility();

    return <IconWrapper onPress={toggleVisibility}>{isVisible ? <Visible /> : <Invisible />}</IconWrapper>;
};
