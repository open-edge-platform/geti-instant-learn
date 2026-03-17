/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, Suspense } from 'react';

import { $api } from '@/api';
import { IntelBrandedLoading, Toast } from '@geti/ui';
import { Outlet } from 'react-router';

import { useAcceptLicense } from '../features/license/api/use-accept-license.hook';
import { License } from '../features/license/license.component';

const HealthCheckup = ({ children }: { children: ReactNode }) => {
    const { data } = $api.useQuery('get', '/health', undefined, {
        refetchInterval: (query) => {
            const healthData = query.state.data;
            return healthData?.status === 'ok' && healthData.license_accepted ? false : 2000;
        },
    });
    const { mutate: acceptLicense, isPending: isAccepting } = useAcceptLicense();

    if (data?.status === 'ok') {
        if (!data.license_accepted) {
            return <License onAccept={() => acceptLicense({})} isAccepting={isAccepting} />;
        }

        return children;
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
