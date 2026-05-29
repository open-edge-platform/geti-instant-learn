/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Page } from '@playwright/test';

export class SinksPage {
    constructor(private page: Page) {}

    async goto() {
        await this.page.goto('/');
    }

    async openPipelineConfiguration() {
        await this.page.getByRole('button', { name: 'Pipeline configuration' }).click();
    }

    async openOutputTab() {
        await this.page.getByRole('tab', { name: 'Output' }).click();
    }

    getSinkCard() {
        return this.page.getByTestId('pipeline-entity-card-mqtt');
    }

    async openSinkActions() {
        await this.getSinkCard().getByRole('button', { name: 'More actions' }).click();
    }

    async selectAction(action: 'Edit' | 'Delete' | 'Connect') {
        await this.page.getByRole('menuitem', { name: action }).click();
    }

    get nameField() {
        return this.page.getByRole('textbox', { name: 'Name' });
    }

    get brokerHostField() {
        return this.page.getByRole('textbox', { name: 'Broker host' });
    }

    get topicField() {
        return this.page.getByRole('textbox', { name: 'Topic' });
    }

    get applyButton() {
        return this.page.getByRole('button', { name: 'Apply' });
    }

    get saveButton() {
        return this.page.getByRole('button', { name: 'Save', exact: true });
    }

    get saveAndConnectButton() {
        return this.page.getByRole('button', { name: 'Save & Connect' });
    }

    get addNewSinkButton() {
        return this.page.getByRole('button', { name: 'Add new sink' });
    }

    get mqttSinkTypeButton() {
        return this.page.getByRole('button', { name: 'MQTT' });
    }

    get editMenuItem() {
        return this.page.getByRole('menuitem', { name: 'Edit' });
    }

    get deleteMenuItem() {
        return this.page.getByRole('menuitem', { name: 'Delete' });
    }

    get connectMenuItem() {
        return this.page.getByRole('menuitem', { name: 'Connect' });
    }
}
