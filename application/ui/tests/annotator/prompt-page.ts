/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Page } from '@playwright/test';

export class PromptPage {
    constructor(private page: Page) {}

    get savePromptButton() {
        return this.page.getByRole('button', { name: 'Save prompt' });
    }

    async savePrompt() {
        await this.savePromptButton.click();
    }
}
