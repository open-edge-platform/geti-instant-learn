/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Page } from '@playwright/test';

export class StreamPage {
    constructor(private page: Page) {}

    async startStream() {
        await this.page.getByRole('button', { name: 'Start Stream' }).click();
    }

    get captureFrameButton() {
        return this.page.getByRole('button', { name: 'Capture' });
    }

    async captureFrame() {
        await this.captureFrameButton.click();
    }

    async stopStream() {
        await this.page.getByRole('button', { name: 'Stop' }).click();
    }
}
