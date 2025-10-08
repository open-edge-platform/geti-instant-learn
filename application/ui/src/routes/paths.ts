/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { path } from 'static-path';

const projects = path('/projects');
const welcome = path('/welcome');
const project = projects.path('/:projectId');

export const paths = {
    root: path('/'),
    welcome,
    projects,
    project,
};
