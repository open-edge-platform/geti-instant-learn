/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { $api } from '@/api';
import { IntelBrandedLoading } from '@geti/ui';

export const HealthCheckup = ({ children }: { children: ReactNode }) => {
    const { data } = $api.useQuery('get', '/health', undefined, {
        refetchInterval: (query) => {
            const healthData = query.state.data;
            return healthData?.status === 'ok' ? false : 2000;
        },
    });

    if (data?.status === 'ok') {
        return children;
    }

    return <IntelBrandedLoading />;
};
