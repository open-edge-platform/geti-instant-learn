/*
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { defineConfig } from '@rsbuild/core';
import { pluginReact } from '@rsbuild/plugin-react';
import { pluginSass } from '@rsbuild/plugin-sass';
import { pluginSvgr } from '@rsbuild/plugin-svgr';

export default defineConfig({
  plugins: [
    pluginReact(),

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
