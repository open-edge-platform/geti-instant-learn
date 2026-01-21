/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

/// <reference types="@rsbuild/core/types" />

declare module '*.svg' {
    export const ReactComponent: React.FunctionComponent<React.SVGProps<SVGSVGElement>>;
}

declare module '*.svg?react' {
    const ReactComponent: React.FunctionComponent<React.SVGProps<SVGSVGElement>>;
    export default ReactComponent;
}

interface ImportMetaEnv {
    // import.meta.env.PUBLIC_API_URL
    readonly PUBLIC_API_URL: string;
}

interface ImportMeta {
    readonly env: ImportMetaEnv;
}

/**
 * Tauri injects the `__TAURI__` object onto `window` when `withGlobalTauri: true`
 * is set in tauri.conf.json. This allows web code to detect if it's running
 * inside a Tauri context and invoke Tauri commands.
 *
 * @see https://v2.tauri.app/reference/config/#withglobaltauri
 */
interface Window {
    __TAURI__?: {
        core?: {
            invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
        };
    };
}
