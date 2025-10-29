/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { LabelType } from '@geti-prompt/api';

export const MAX_LABEL_NAME_LENGTH = 100;

export const isUniqueLabelName = (name: string, existingLabels: LabelType[], excludeId?: string): boolean => {
    return !existingLabels.some((label) => label.name === name && label.id !== excludeId);
};

export const validateLabelName = (name: string, existingLabels: LabelType[], excludeId?: string): string | null => {
    const trimmedName = name.trim();

    if (!trimmedName) {
        return 'Label name cannot be empty.';
    }

    if (!isUniqueLabelName(trimmedName, existingLabels, excludeId)) {
        return 'Label name must be unique.';
    }

    return null;
};
