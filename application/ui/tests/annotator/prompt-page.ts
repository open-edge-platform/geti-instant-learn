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

    get thumbnails() {
        return this.page.getByLabel(/prompt thumbnail/i);
    }

    async savePrompt() {
        await this.savePromptButton.click();
    }

    async deletePrompt(id: string) {
        await this.page.getByLabel(`prompt thumbnail ${id}`).getByLabel('Prompt actions').click();
    }
}
