/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ProjectType } from '@/api';

export const getMockedProject = (customProject: Partial<ProjectType>): ProjectType => {
    return {
        id: '7b073838-99d3-42ff-9018-4e901eb047fc',
        name: 'animals',
        active: false,
        ...customProject,
    };
};
