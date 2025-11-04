/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Page } from '@playwright/test';

import { paths } from '../../src/routes/paths';

export class ProjectPage {
    constructor(private page: Page) {}

    async goto(projectId: string) {
        await this.page.goto(paths.project({ projectId }));
    }

    async gotoProjects() {
        await this.page.goto(paths.projects({}));
    }

    get welcomeHeader() {
        return this.page.getByRole('heading', { name: 'Welcome to Geti Prompt' });
    }

    get projectsHeader() {
        return this.page.getByRole('heading', { name: 'Projects' });
    }

    async create() {
        await this.page.getByRole('button', { name: 'Create project' }).click();
    }

    get createProjectConfirmationDialogHeading() {
        return this.page.getByRole('heading', { name: 'Create project' });
    }

    async createConfirmation() {
        await this.page.getByRole('button', { name: 'Create' }).click();
    }

    getProjectInTheList(projectName: string) {
        return this.page.getByRole('listitem', { name: `Project ${projectName}` });
    }

    getSelectedProject(projectName: string) {
        return this.page.getByRole('button', { name: `Selected project ${projectName}` });
    }

    async openProjectManagementPanel() {
        await this.page.getByRole('button', { name: /Selected project/ }).click();
    }

    async openProjectMenu(projectName: string) {
        await this.page.getByLabel(`Project ${projectName}`).getByRole('button', { name: 'Project actions' }).click();
    }

    async selectMenuItem(itemName: 'Rename' | 'Delete' | 'Activate' | 'Deactivate') {
        await this.page.getByRole('menuitem', { name: itemName }).click();
    }

    async updateProjectName(newName: string) {
        await this.page.getByRole('textbox', { name: 'Edit project name' }).fill(newName);
        await this.page.getByRole('textbox', { name: 'Edit project name' }).press('Enter');
    }

    async delete() {
        await this.page.getByRole('button', { name: 'Delete' }).click();
    }

    async activate() {
        await this.page.getByRole('button', { name: 'Activate' }).click();
    }

    async activateCurrentProject() {
        await this.page.getByRole('button', { name: 'Activate current project' }).click();
    }

    async deactivateCurrentProject() {
        await this.page.getByRole('button', { name: 'Deactivate current project' }).click();
    }

    get inactiveStatus() {
        return this.page.getByLabel(/Selected project/).getByLabel('Inactive project');
    }

    get activeStatus() {
        return this.page.getByLabel(/Selected project/).getByLabel('Active project');
    }

    getActiveProjectInTheList(projectName: string) {
        return this.page.getByRole('listitem', { name: `Project ${projectName}` }).getByLabel('Active project');
    }

    getInactiveProjectInTheList(projectName: string) {
        return this.page.getByRole('listitem', { name: `Project ${projectName}` }).getByLabel('Inactive project');
    }

    get activateProjectDialogHeading() {
        return this.page.getByTestId('modal').getByRole('heading', { name: 'Activate project' });
    }
}
