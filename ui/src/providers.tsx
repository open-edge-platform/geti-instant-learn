/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Suspense } from 'react';

import { IntelBrandedLoading } from '@geti/ui';
import { ThemeProvider } from '@geti/ui/theme';
import { MutationCache, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from 'react-router';

import { router } from './routes/router';

export const queryClient = new QueryClient({
    mutationCache: new MutationCache({
        onSuccess: () => {
            queryClient.invalidateQueries();
        },
    }),
});

export const Providers = () => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider router={router}>
                <Suspense fallback={<IntelBrandedLoading />}>
                    <RouterProvider router={router} />
                </Suspense>
            </ThemeProvider>
        </QueryClientProvider>
    );
};
