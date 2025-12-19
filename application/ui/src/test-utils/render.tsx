/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Suspense, type ReactNode } from 'react';

import { queryClient } from '@geti-prompt/query-client';
import { IntelBrandedLoading, Toast } from '@geti/ui';
import { ThemeProvider } from '@geti/ui/theme';
import { QueryClientProvider } from '@tanstack/react-query';
import {
    render as rtlRender,
    renderHook as rtlRenderHook,
    RenderOptions as RTLRenderOptions,
} from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';

import { paths } from '../constants/paths';

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

const createTestRouter = (children: ReactNode, options: RenderOptions) => {
    return createMemoryRouter(
        [
            {
                path: options.path,
                element: <TestProviders>{children}</TestProviders>,
            },
        ],
        {
            initialEntries: [options.route],
            initialIndex: 0,
        }
    );
};

export const render = (
    ui: ReactNode,
    options: RenderOptions = { route: paths.project({ projectId: '1' }), path: paths.project.pattern }
) => {
    const router = createTestRouter(ui, options);

    return rtlRender(<RouterProvider router={router} />);
};

export const renderHook = <TProps, TResult>(
    callback: (props: TProps) => TResult,
    options: RenderOptions = { route: paths.project({ projectId: '1' }), path: paths.project.pattern }
) => {
    const Wrapper = ({ children }: { children: ReactNode }) => {
        const router = createTestRouter(children, options);

        return <RouterProvider router={router} />;
    };

    return rtlRenderHook(callback, { wrapper: Wrapper });
};
