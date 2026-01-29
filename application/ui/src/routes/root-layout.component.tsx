/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, Suspense, useState } from 'react';

import { $api } from '@/api';
import { IntelBrandedLoading, Toast } from '@geti/ui';
import { Outlet } from 'react-router';

import { License } from '../features/license/license.component';

/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
const HealthCheckup = ({ children }: { children: ReactNode }) => {
    const { data } = $api.useQuery('get', '/health', undefined, {
        refetchInterval: (query) => {
            return query.state.data?.status === 'ok' ? false : 2000;
        },
    });
    const [licenseAccepted, setLicenseAccepted] = useState<boolean>(false);

    if (data?.status === 'ok') {
        /*if (licenseAccepted) {
            return children
        }*/

        return (
            <>
                <License isVisible={!licenseAccepted} onAccept={() => setLicenseAccepted(true)} />
                {licenseAccepted && children}
            </>
        );
    }

    return <IntelBrandedLoading />;
};

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
