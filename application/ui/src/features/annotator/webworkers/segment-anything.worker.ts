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

        console.log('[SAM worker] ort.env.wasm.wasmPaths:', ort.env.wasm.wasmPaths);
        console.log('[SAM worker] ort.env.wasm.numThreads:', ort.env.wasm.numThreads);

        return proxy({
            init: (algorithmType: Parameters<typeof instance.init>[0]) => {
                console.log('[SAM worker] init:', algorithmType);
                return instance.init(algorithmType);
            },
            processEncoder: async (imageData: ImageData) => {
                console.log('[SAM worker] processEncoder start, image:', imageData?.width, 'x', imageData?.height);
                try {
                    const result = await instance.processEncoder(imageData);
                    const r = result.encoderResult as unknown as Record<string, unknown>;
                    console.log('[SAM worker] processEncoder OK - encoderResult.type:', r?.type,
                        'dims:', r?.dims,
                        'data constructor:', (r?.data as { constructor?: { name: unknown } })?.constructor?.name,
                        'hasGetData:', typeof r?.getData);
                    return result;
                } catch (err) {
                    console.error('[SAM worker] processEncoder threw:', err);
                    throw err;
                }
            },
            processDecoder: (
                encoding: Parameters<typeof instance.processDecoder>[0],
                input: Parameters<typeof instance.processDecoder>[1]
            ) => {
                const enc = encoding.encoderResult as unknown as Record<string, unknown>;
                console.log('[SAM worker] processDecoder - encoderResult.type:', enc?.type,
                    'dims:', enc?.dims,
                    'hasGetData:', typeof enc?.getData);
                return instance.processDecoder(encoding, input);
            },
        });
    },
};

expose(WorkerApi);
