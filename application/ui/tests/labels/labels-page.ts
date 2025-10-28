/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Page } from '@playwright/test';

export class LabelsPage {
    constructor(private page: Page) {}

    async showDialog() {
        await this.page.getByRole('button', { name: 'Add Label' }).click();
    }

    async enterName(name: string) {
        await this.page.getByRole('textbox', { name: 'New label name' }).fill(name);
    }

    getConfirmLabel() {
        return this.page.getByRole('button', { name: 'Confirm label' });
    }

    confirmLabel() {
        return this.getConfirmLabel().click();
    }

    getLabel(name: string) {
        return this.page.getByLabel(`Label ${name}`);
    }

    async deleteLabel(name: string) {
        await this.getLabel(name).hover();

        await this.getLabel(name)
            .getByRole('button', { name: `Delete ${name} label` })
            .click();
    }
}
