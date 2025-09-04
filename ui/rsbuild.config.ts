/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { defineConfig } from '@rsbuild/core';
import { pluginBabel } from '@rsbuild/plugin-babel';
import { pluginReact } from '@rsbuild/plugin-react';
import { pluginSass } from '@rsbuild/plugin-sass';
import { pluginSvgr } from '@rsbuild/plugin-svgr';

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

    html: {
        title: 'Geti Prompt',
    },
});
