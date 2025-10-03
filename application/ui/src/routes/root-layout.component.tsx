/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Suspense } from 'react';

import { IntelBrandedLoading } from '@geti/ui';
import { Outlet } from 'react-router';

export const RootLayout = () => {
    return (
        <Suspense fallback={<IntelBrandedLoading />}>
            <Outlet />
        </Suspense>
    );
};
