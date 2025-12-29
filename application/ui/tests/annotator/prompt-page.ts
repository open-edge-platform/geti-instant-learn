/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Page } from '@playwright/test';

export class PromptPage {
    constructor(
        private readonly page: Page,
        private readonly scope = page.locator('body')
    ) {}

    get savePromptButton() {
        return this.scope.getByRole('button', { name: 'Save prompt' });
    }

    get thumbnail() {
        return this.page.getByLabel(/prompt thumbnail/i);
    }

    async savePrompt() {
        await this.savePromptButton.click();
    }

    async deletePrompt(promptId: string) {
        const thumbnail = this.page.getByLabel(`prompt thumbnail ${promptId}`);
        await thumbnail.hover();

        await this.page.getByLabel(`Prompt actions ${promptId}`).click();
        await this.page.getByText('Delete').click();
    }

    async editPrompt(promptId: string) {
        const thumbnail = this.page.getByLabel(`prompt thumbnail ${promptId}`);
        await thumbnail.hover();

        await this.page.getByLabel(`Prompt actions ${promptId}`).click();
        await this.page.getByText('Edit').click();
    }

    getCapturedFrame(frameId: string) {
        return this.scope.getByTestId(`captured-frame-${frameId}`);
    }
}
