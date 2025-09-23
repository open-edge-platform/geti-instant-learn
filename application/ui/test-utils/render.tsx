/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { type ReactNode, Suspense } from 'react';

import { IntelBrandedLoading } from '@geti/ui';
import { ThemeProvider } from '@geti/ui/theme';
import { QueryClientProvider } from '@tanstack/react-query';
import { RenderOptions, render as rtlRender } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';

import { queryClient } from '../src/providers';
import { paths } from '../src/routes/paths';

interface Options extends RenderOptions {
    route: string;
    path: string;
}

const TestProviders = ({ children }: { children: ReactNode }) => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider>
                <Suspense fallback={<IntelBrandedLoading />}>{children}</Suspense>
            </ThemeProvider>
        </QueryClientProvider>
    );
};

export const render = (
    ui: ReactNode,
    options: Options = { route: paths.project({ projectId: '1' }), path: paths.project.pattern }
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
