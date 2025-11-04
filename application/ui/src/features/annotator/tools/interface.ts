/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ComponentType, SVGProps } from 'react';

export type ToolType = 'sam';

export interface ToolConfig {
    type: ToolType;
    icon: ComponentType<SVGProps<SVGSVGElement>>;
}
