/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, type Locator, type Page } from '@playwright/test';

export class AnnotatorPage {
    constructor(
        private readonly page: Page,
        protected readonly scope: Locator = page.locator('body')
    ) {}

    getScope() {
        return this.scope;
    }

    getFullScreen() {
        return new AnnotatorPageFullScreen(this.page, this.page.getByRole('dialog'));
    }

    getCapturedFrame() {
        return this.scope.getByLabel('Captured frame');
    }

    getProcessingImage() {
        return this.scope.getByText('Processing image, please wait...');
    }

    async annotateAt(x: number, y: number) {
        const image = this.getCapturedFrame();
        const box = await image.boundingBox();

        if (box) {
            const hoverX = x;
            const hoverY = y;

            // Hover to trigger preview
            await this.page.mouse.move(hoverX, hoverY);

            // Wait for preview to appear
            await expect(this.scope.getByLabel('Segment anything preview')).toBeVisible({ timeout: 10000 });

            await this.page.mouse.click(hoverX, hoverY);
        }
    }

    async addAnnotation() {
        const image = this.getCapturedFrame();
        const box = await image.boundingBox();

        if (box) {
            // Position: middle horizontally, 20% from the bottom vertically
            const hoverX = box.x + box.width / 2;
            const hoverY = box.y + box.height * 0.8;

            await this.annotateAt(hoverX, hoverY);
        }
    }

    getAnnotation() {
        return this.scope.getByLabel('annotation list').getByLabel('annotation polygon');
    }

    async hideAnnotations() {
        await this.scope.getByRole('button', { name: 'Hide annotations' }).click();
    }

    async showAnnotations() {
        await this.scope.getByRole('button', { name: 'Show annotations' }).click();
    }

    async undoAnnotation() {
        await this.scope.getByRole('button', { name: 'undo' }).click();
    }

    async redoAnnotation() {
        await this.scope.getByRole('button', { name: 'redo' }).click();
    }

    async openFullscreen() {
        await this.scope.getByRole('button', { name: 'Open full screen' }).click();

        return new AnnotatorPageFullScreen(this.page, this.page.getByRole('dialog'));
    }

    async openSettings() {
        await this.scope.getByRole('button', { name: 'Settings' }).click();
    }

    async closeSettings() {
        await this.scope.getByRole('button', { name: 'Close settings' }).click();
    }

    async zoomIn() {
        await this.scope.getByRole('button', { name: 'Zoom in' }).click();
    }

    async zoomOut() {
        await this.scope.getByRole('button', { name: 'Zoom out' }).click();
    }

    async fitToScreen() {
        await this.scope.getByRole('button', { name: 'Fit image to screen' }).click();
    }

    async getZoomValue() {
        return this.scope.getByTestId('zoom-level');
    }
}

class AnnotatorPageFullScreen extends AnnotatorPage {
    constructor(page: Page, scope: Locator) {
        super(page, scope);
    }

    async close() {
        await this.scope.getByRole('button', { name: 'Close full screen' }).click();
    }
}
