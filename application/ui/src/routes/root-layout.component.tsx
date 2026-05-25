/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Suspense } from 'react';

import { IntelBrandedLoading, Toast } from '@geti/ui';
import { Outlet } from 'react-router';

import { HealthCheckup } from './health-checkup.component';

export const RootLayout = () => {
    return (
        <Suspense fallback={<IntelBrandedLoading />}>
            <HealthCheckup>
                <Outlet />
                <Toast />
            </HealthCheckup>
        </Suspense>
    );
};
