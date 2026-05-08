/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { buildSegmentAnythingInstance } from '@geti/smart-tools/segment-anything';
import { expose, proxy } from 'comlink';
import * as ort from 'onnxruntime-web';

// WASM files are copied to /ort-wasm/ at build time (rsbuild.config.ts output.copy).
// The package's wasm-utils.ts is patched by scripts/patch-wasm-utils.js (postinstall) to use
// wasmRoot: '/ort-wasm/' and numThreads: 1 directly, avoiding broken new URL() resolution
// in Tauri/WebView2 workers.
//
// Object.defineProperty below acts as a safety net in case the patch didn't run — it prevents
// session.ts from overriding env.wasm properties with incorrect values.
Object.defineProperty(ort.env.wasm, 'wasmPaths', {
    get: () => '/ort-wasm/',
    set: () => {
        /* safety net: ignore any override from the package */
    },
    configurable: true,
});
Object.defineProperty(ort.env.wasm, 'numThreads', {
    get: () => 1,
    set: () => {
        /* safety net: single-threaded mode for Tauri/WebView2 */
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
