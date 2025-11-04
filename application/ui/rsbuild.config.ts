/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { defineConfig, loadEnv } from '@rsbuild/core';
import { pluginBabel } from '@rsbuild/plugin-babel';
import { pluginReact } from '@rsbuild/plugin-react';
import { pluginSass } from '@rsbuild/plugin-sass';
import { pluginSvgr } from '@rsbuild/plugin-svgr';

const { publicVars } = loadEnv();

export default defineConfig({
    plugins: [
        pluginReact(),

        // Enables React Compiler
        pluginBabel({
            include: /\.(?:jsx|tsx)$/,
            babelLoaderOptions(opts) {
                opts.plugins?.unshift('babel-plugin-react-compiler');
            },
        }),

        pluginSass(),

        pluginSvgr({
            svgrOptions: {
                exportType: 'named',
            },
        }),
    ],
    output: {
        assetPrefix: process.env.ASSET_PREFIX,
    },
    source: {
        define: {
            ...publicVars,
            'import.meta.env.PUBLIC_API_URL': publicVars['import.meta.env.PUBLIC_API_URL'] ?? '"http://localhost:9100"',
            'process.env.PUBLIC_API_URL': publicVars['import.meta.env.PUBLIC_API_URL'] ?? '"http://localhost:9100"',
            // Needed to prevent an issue with spectrum's picker
            // eslint-disable-next-line max-len
            // https://github.com/adobe/react-spectrum/blob/6173beb4dad153aef74fc81575fd97f8afcf6cb3/packages/%40react-spectrum/overlays/src/OpenTransition.tsx#L40
            'process.env': {},
        },
    },

    html: {
        title: 'Geti Prompt',
    },

    server: {
        headers: {
            'Cross-Origin-Embedder-Policy': 'credentialless',
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Content-Security-Policy':
                "default-src 'self'; " +
                "script-src 'self' 'unsafe-eval' blob:; " +
                "worker-src 'self' blob:; " +
                "connect-src 'self' http://localhost:9100 data:; " +
                "img-src 'self' http://localhost:9100 data: blob:; " +
                "style-src 'self' 'unsafe-inline';",
        },
    },
});
