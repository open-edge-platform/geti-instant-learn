/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from 'react';

import { ThemeProvider } from '@geti/ui/theme';

export const Providers = ({ children }: { children: ReactNode }) => {
    return <ThemeProvider>{children}</ThemeProvider>;
};
