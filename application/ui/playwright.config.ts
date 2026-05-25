/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'path';

import { defineConfig, devices } from '@playwright/test';
import dotenv from 'dotenv';

const CI = !!process.env.CI;

const file = fileURLToPath(import.meta.url);
const dirname = path.dirname(file);

dotenv.config({
    path: path.resolve(dirname, '.env.test'),
});

// In CI we serve pre-built bundles via `rsbuild preview`, which requires the
// output directories to already exist. Failing fast here produces a clearer
// error than waiting for `webServer.timeout` to elapse on a 404-returning
// preview server.
if (CI) {
    const requiredDirs = ['dist', 'dist-tauri'];
    for (const dir of requiredDirs) {
        const absolute = path.resolve(dirname, dir);
        if (!existsSync(absolute)) {
            throw new Error(
                `Missing build output at ${absolute}. ` +
                    `Run \`npm run build\` (web) and \`npm run build:tauri\` (tauri) before \`npm run test:component\`.`
            );
        }
    }
}

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
    testDir: './tests',
    /* Run tests in files in parallel */
    fullyParallel: true,
    /* Fail the build on CI if you accidentally left test.only in the source code. */
    forbidOnly: CI,
    /* Retry on CI only */
    retries: process.env.CI ? 2 : 0,
    /* Opt out of parallel tests on CI. */
    workers: process.env.CI ? 1 : undefined,
    /* Reporter to use. See https://playwright.dev/docs/test-reporters */
    reporter: [[CI ? 'github' : 'list'], ['html', { open: 'never' }]],
    use: {
        baseURL: 'http://localhost:3000',
        trace: CI ? 'on-first-retry' : 'on',
        video: CI ? 'on-first-retry' : 'on',
        launchOptions: {
            slowMo: 100,
            headless: true,
            devtools: !CI,
        },
        viewport: { height: 1480, width: 1920 },
        timezoneId: 'UTC',
        actionTimeout: CI ? 10000 : 5000,
        navigationTimeout: CI ? 10000 : 5000,
    },

    /* Configure projects for major browsers */
    projects: [
        {
            name: 'Component tests',
            use: { ...devices['Desktop Chrome'] },
            testIgnore: /.*\.tauri\.spec\.ts$/,
        },
        {
            // Tauri-flavored specs run against a bundle built with
            // `BUILD_TARGET=tauri`, which substitutes `*.tauri.tsx` overrides
            // for their web counterparts (see rsbuild.config.ts). They are
            // served on a separate port so both bundles can coexist.
            name: 'Tauri component tests',
            use: { ...devices['Desktop Chrome'], baseURL: 'http://localhost:3001' },
            testMatch: /.*\.tauri\.spec\.ts$/,
        },
    ],

    /* Run your local dev server before starting the tests */
    webServer: [
        {
            command: CI ? 'npm run preview' : 'npm start',
            name: 'client',
            url: 'http://localhost:3000',
            reuseExistingServer: CI === false,
        },
        {
            command: CI ? 'npm run preview:tauri' : 'npm run start:tauri -- --port 3001',
            name: 'tauri-client',
            url: 'http://localhost:3001',
            reuseExistingServer: CI === false,
        },
    ],
});
