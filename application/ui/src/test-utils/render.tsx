/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Suspense, type ReactNode } from 'react';

import { IntelBrandedLoading, Toast } from '@geti/ui';
import { ThemeProvider } from '@geti/ui/theme';
import { QueryClientProvider } from '@tanstack/react-query';
import { render as rtlRender, RenderOptions as RTLRenderOptions } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';

import { queryClient } from '../query-client/query-client';
import { paths } from '../routes/paths';

export interface RenderOptions extends RTLRenderOptions {
    route: string;
    path: string;
}

const TestProviders = ({ children }: { children: ReactNode }) => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider>
                <Suspense fallback={<IntelBrandedLoading />}>{children}</Suspense>
                <Toast />
            </ThemeProvider>
        </QueryClientProvider>
    );
};

export const render = (
    ui: ReactNode,
    options: RenderOptions = { route: paths.project({ projectId: '1' }), path: paths.project.pattern }
) => {
    const router = createMemoryRouter(
        [
            {
                path: options.path,
                element: <TestProviders>{ui}</TestProviders>,
            },
        ],
        {
            initialEntries: [options.route],
            initialIndex: 0,
        }
    );

    return rtlRender(<RouterProvider router={router} />);
};
