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
        await this.page.getByRole('textbox', { name: 'Label name' }).fill(name);
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

    async enterEditLabelMode(name: string) {
        await this.getLabel(name).hover();

        await this.getLabel(name)
            .getByRole('button', { name: `Edit ${name} label` })
            .click();
    }

    async updateLabelName(oldName: string, newName: string) {
        await this.enterEditLabelMode(oldName);

        await this.enterName(newName);
        await this.confirmLabel();
    }

    getColorPickerButton() {
        return this.page.getByRole('button', { name: 'Color picker button' });
    }

    getColorInput() {
        return this.page.getByTestId('change-color-button-color-input');
    }

    openColorPicker() {
        return this.getColorPickerButton().click();
    }

    async changeColor() {
        const colorPickerArea = this.page.getByLabel('Color', { exact: true });

        const colorPickerBoundingBox = await colorPickerArea.boundingBox();

        if (colorPickerBoundingBox === null) {
            return;
        }

        await this.page.mouse.click(colorPickerBoundingBox.x + 10, colorPickerBoundingBox.y + 10);
    }
}
