/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Patches packages/smart-tools/src/utils/wasm-utils.ts after npm install clones it.
 *
 * The original wasm-utils.ts uses new URL(..., import.meta.url) to resolve WASM paths,
 * which breaks inside Web Workers in Tauri/WebView2 (import.meta.url resolves to a
 * blob: URL, making relative paths resolve incorrectly).
 *
 * This patch replaces the broken path resolution with a static prefix ('/ort-wasm/')
 * and forces single-threaded mode to avoid SharedArrayBuffer thread worker issues.
 *
 * The WASM files are copied to dist/ort-wasm/ at build time via rsbuild.config.ts output.copy.
 */

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const wasmUtilsPath = resolve(__dirname, '../packages/smart-tools/src/utils/wasm-utils.ts');

if (!existsSync(wasmUtilsPath)) {
    console.warn(`[patch-wasm-utils] File not found: ${wasmUtilsPath} — skipping patch.`);
    process.exit(0);
}

const original = readFileSync(wasmUtilsPath, 'utf-8');

// Check if already patched to avoid double-patching
if (original.includes('// patched-for-tauri')) {
    console.log('[patch-wasm-utils] Already patched, skipping.');
    process.exit(0);
}

// Replace the broken new URL() wasm path resolution and numThreads: 0 with Tauri-compatible values.
// WASM files are served from /ort-wasm/ (copied at build time by rsbuild.config.ts output.copy).
const patched = `// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE
// patched-for-tauri: wasm-utils.ts patched by scripts/patch-wasm-utils.js

export interface SessionParameters {
    numThreads: number;
    executionProviders: string[];
    wasmRoot?: string | Record<string, string>;
}

// Use a static prefix so WASM files are fetched from /ort-wasm/ at runtime.
// WASM files are copied there at build time via rsbuild.config.ts output.copy.
// new URL(..., import.meta.url) breaks inside workers in Tauri/WebView2.
export const sessionParams: SessionParameters = {
    numThreads: 1,
    executionProviders: ['cpu'],
    wasmRoot: '/ort-wasm/',
};
`;

writeFileSync(wasmUtilsPath, patched, 'utf-8');
console.log(`[patch-wasm-utils] Patched ${wasmUtilsPath}`);
