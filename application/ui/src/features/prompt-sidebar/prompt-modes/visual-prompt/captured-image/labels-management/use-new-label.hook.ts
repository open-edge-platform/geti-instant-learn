/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { getDistinctColorBasedOnHash } from '@geti/ui/utils';
import { v4 as uuid } from 'uuid';

import { Label } from './label.interface';

export const useNewLabel = () => {
    const getBaseNewLabel = (): Label => {
        const newId = uuid();
        return { id: newId, name: '', color: getDistinctColorBasedOnHash(newId) };
    };

    const [label, setLabel] = useState<Label>(getBaseNewLabel());

    const resetState = () => {
        setLabel(getBaseNewLabel());
    };

    return { resetState, label };
};
