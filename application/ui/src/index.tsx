/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';

import { createRoot } from 'react-dom/client';

import { Providers } from './providers';

import './index.scss';

const rootEl = document.getElementById('root');

if (rootEl) {
    const root = createRoot(rootEl);

    root.render(
        <React.StrictMode>
            <Providers />
        </React.StrictMode>
    );
}
