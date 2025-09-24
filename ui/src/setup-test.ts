/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import '@testing-library/jest-dom/vitest';

import { setupServer } from 'msw/node';
import fetchPolyfill, { Request as RequestPolyfill } from 'node-fetch';
import { afterAll, afterEach, beforeAll } from 'vitest';

import { handlers, http } from './api/utils';

// Initialize msw's mock server with the handlers
const server = setupServer(...handlers);

export { server, http };

beforeAll(() => {
    server.listen({ onUnhandledRequest: 'bypass' });
});

afterEach(() => {
    server.resetHandlers();
});

afterAll(() => {
    server.close();
});

// Why we need these polyfills:
// https://github.com/reduxjs/redux-toolkit/issues/4966#issuecomment-3115230061
Object.defineProperty(global, 'fetch', {
    // MSW will overwrite this to intercept requests
    writable: true,
    value: fetchPolyfill,
});

Object.defineProperty(global, 'Request', {
    writable: false,
    value: RequestPolyfill,
});
