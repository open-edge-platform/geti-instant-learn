/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useSearchParams } from 'react-router-dom';

type PromptMode = 'visual' | 'text';

export const usePromptMode = (): PromptMode => {
    const [searchParams] = useSearchParams();

    return (searchParams.get('mode') as PromptMode) ?? 'visual';
};
