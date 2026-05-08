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
    set: (v) => {
        console.warn('[SAM worker] wasmPaths write intercepted, ignoring value:', v);
    },
    configurable: true,
});
// Single-threaded mode: nested SharedArrayBuffer workers can hang in WebView2
// even when crossOriginIsolated is true.
Object.defineProperty(ort.env.wasm, 'numThreads', {
    get: () => 1,
    set: (v) => {
        console.warn('[SAM worker] numThreads write intercepted, ignoring value:', v);
    },
    configurable: true,
});

// Diagnostic: log the actual env.wasm state after module init
console.log('[SAM worker] ort.env.wasm after defineProperty:', JSON.stringify({
    wasmPaths: ort.env.wasm.wasmPaths,
    numThreads: ort.env.wasm.numThreads,
    simd: ort.env.wasm.simd,
    proxy: ort.env.wasm.proxy,
}));

// Diagnostic: intercept InferenceSession.create to log actual options
const _origCreate = ort.InferenceSession.create.bind(ort.InferenceSession);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(ort.InferenceSession as any).create = async function (...args: unknown[]) {
    console.log('[SAM worker] InferenceSession.create called, options:', JSON.stringify(args[1]));
    console.log('[SAM worker] env.wasm at create time:', JSON.stringify({
        wasmPaths: ort.env.wasm.wasmPaths,
        numThreads: ort.env.wasm.numThreads,
    }));
    return _origCreate(...(args as Parameters<typeof ort.InferenceSession.create>));
};

const WorkerApi = {
    build: async () => {
        const instance = await buildSegmentAnythingInstance();

        return proxy(instance);
    },
};

expose(WorkerApi);
