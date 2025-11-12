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

    get thumbnail() {
        return this.page.getByLabel(/prompt thumbnail/i);
    }

    async savePrompt() {
        await this.savePromptButton.click();
    }

    async deletePrompt() {
        await this.page.getByLabel(/prompt thumbnail/i).hover();
        await this.page.getByLabel('Prompt actions').click();
        await this.page.getByText('Delete').click();
    }
}
