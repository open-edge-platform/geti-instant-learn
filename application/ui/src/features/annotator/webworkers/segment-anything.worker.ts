/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { buildSegmentAnythingInstance } from '@geti/smart-tools/segment-anything';
import { expose, proxy } from 'comlink';
import * as ort from 'onnxruntime-web';

// WASM files are copied to /ort-wasm/ at build time (rsbuild.config.ts output.copy).
// Override here because new URL(..., import.meta.url) in the package resolves incorrectly
// inside Web Workers in Tauri/WebView2, returning an HTML 404 page instead of the binary.
ort.env.wasm.wasmPaths = '/ort-wasm/';
// Single-threaded mode: nested SharedArrayBuffer workers can hang in WebView2
// even when crossOriginIsolated is true.
ort.env.wasm.numThreads = 1;

const WorkerApi = {
    build: async () => {
        const instance = await buildSegmentAnythingInstance();

        return proxy(instance);
    },
};

expose(WorkerApi);
