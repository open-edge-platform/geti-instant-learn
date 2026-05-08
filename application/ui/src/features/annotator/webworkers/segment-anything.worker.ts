/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { buildSegmentAnythingInstance } from '@geti/smart-tools/segment-anything';
import { expose, proxy } from 'comlink';
import * as ort from 'onnxruntime-web';

// WASM files are copied to /ort-wasm/ at build time (rsbuild.config.ts output.copy).
// The package's wasm-utils.ts uses new URL(..., import.meta.url) which resolves incorrectly
// inside Web Workers in Tauri/WebView2, returning an HTML 404 page instead of the binary.
// We use Object.defineProperty to prevent session.ts from overriding our path with the broken one.
Object.defineProperty(ort.env.wasm, 'wasmPaths', {
    get: () => '/ort-wasm/',
    set: () => {
        /* ignore overrides from the package */
    },
    configurable: true,
});
// Single-threaded mode: nested SharedArrayBuffer workers can hang in WebView2
// even when crossOriginIsolated is true.
Object.defineProperty(ort.env.wasm, 'numThreads', {
    get: () => 1,
    set: () => {
        /* ignore overrides from the package */
    },
    configurable: true,
});

const WorkerApi = {
    build: async () => {
        const instance = await buildSegmentAnythingInstance();

        return proxy(instance);
    },
};

expose(WorkerApi);
