/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ProjectType } from '@geti-prompt/api';

export type ProjectWithActiveStatus = ProjectType & { isActive: boolean };
