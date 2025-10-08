/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { MutationCache, QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
    mutationCache: new MutationCache({
        onSuccess: () => {
            queryClient.invalidateQueries();
        },
    }),
});
