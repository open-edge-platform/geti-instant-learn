/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, Page } from '@playwright/test';

export class AnnotatorPage {
    constructor(private page: Page) {}

    async startSAM() {
        await this.page.getByRole('button', { name: 'Select SAM Tool' }).click();
    }

    getCapturedFrame() {
        return this.page.getByAltText('Captured frame');
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
            await expect(this.page.getByLabel('Segment anything preview')).toBeVisible({ timeout: 10000 });

            await this.page.mouse.click(hoverX, hoverY);

            // One for the annotation, and the other for the preview.
            await expect(this.page.getByLabel('annotation polygon')).toHaveCount(2);
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
}
