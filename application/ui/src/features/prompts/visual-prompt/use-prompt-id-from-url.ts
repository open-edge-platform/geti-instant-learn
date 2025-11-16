/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useSearchParams } from 'react-router-dom';

export const usePromptIdFromUrl = () => {
    const [searchParams, setSearchParams] = useSearchParams();

    const promptId = searchParams.get('promptId');

    const setPromptId = (id: string | null) => {
        const newParams = new URLSearchParams(searchParams);

        if (id === null) {
            newParams.delete('promptId');
        } else {
            newParams.set('promptId', id);
        }

        setSearchParams(newParams);
    };

    return {
        promptId,
        setPromptId,
    };
};
