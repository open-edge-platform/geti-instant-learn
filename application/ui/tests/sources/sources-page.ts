/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Page } from '@playwright/test';

export class SourcesPage {
    constructor(private page: Page) {}

    async goto() {
        await this.page.goto('/');
    }

    async openPipelineConfiguration() {
        await this.page.getByRole('button', { name: 'Pipeline configuration' }).click();
    }

    async openAddSourcePanel() {
        const addNewSource = this.page.getByRole('button', { name: 'Add new source' });
        if (await addNewSource.isVisible()) {
            await addNewSource.click();
        }
    }

    async openSourceTypePanel(type: string) {
        await this.openAddSourcePanel();
        await this.page.getByRole('button', { name: type }).click();
    }

    getSourceCard(cardTitle: string) {
        return this.page.getByTestId(`pipeline-entity-card-${cardTitle.toLowerCase().replace(/\s+/g, '-')}`);
    }

    async openSourceActions(cardTitle: string) {
        await this.getSourceCard(cardTitle).getByRole('button', { name: 'More actions' }).click();
    }

    async selectAction(action: 'Edit' | 'Delete' | 'Connect') {
        await this.page.getByRole('menuitem', { name: action }).click();
    }

    get addNewSourceButton() {
        return this.page.getByRole('button', { name: 'Add new source' });
    }
}
