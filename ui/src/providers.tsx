/*
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ThemeProvider } from '@geti/ui/theme';
import type { ReactNode } from 'react';

export const Providers = ({ children }: { children: ReactNode }) => {
  return <ThemeProvider>{children}</ThemeProvider>;
};
