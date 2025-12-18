/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Locator, Page } from '@playwright/test';

export class LabelsPage {
    constructor(
        private readonly page: Page,
        private readonly scope: Locator = page.locator('body')
    ) {}

    async showDialog() {
        await this.scope.getByRole('button', { name: 'Add Label' }).click();
    }

    private async enterName(name: string) {
        await this.page.getByRole('textbox', { name: 'New label name' }).fill(name);
    }

    private getConfirmLabel() {
        return this.page.getByRole('button', { name: 'Confirm label' });
    }

    private confirmLabel() {
        return this.getConfirmLabel().click();
    }

    getLabel(name: string) {
        return this.scope.getByLabel(`Label ${name}`);
    }

    async addLabel(name: string) {
        await this.enterName(name);
        await this.confirmLabel();
    }

    async deleteLabel(name: string) {
        await this.getLabel(name).hover();

        await this.getLabel(name)
            .getByRole('button', { name: `Delete ${name} label` })
            .click();
    }

    async updateLabelName(oldName: string, newName: string) {
        await this.getLabel(oldName).hover();

        await this.getLabel(oldName)
            .getByRole('button', { name: `Edit ${oldName} label` })
            .click();

        await this.enterName(newName);
        await this.confirmLabel();
    }
}
