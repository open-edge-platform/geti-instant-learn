/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, http, test } from '@geti-prompt/test-fixtures';
import { NetworkFixture } from '@msw/playwright';
import { Page } from '@playwright/test';

import { ProjectType } from '../src/api';
import { paths } from '../src/routes/paths';

const registerApiProjects = ({
    network,
    defaultProjects = [],
}: {
    network: NetworkFixture;
    defaultProjects: ProjectType[];
}) => {
    let projects: ProjectType[] = [...defaultProjects];

    network.use(
        http.get('/api/v1/projects', ({ response }) =>
            response(200).json({
                projects,
                pagination: {
                    count: projects.length,
                    total: projects.length,
                    offset: 0,
                    limit: 10,
                },
            })
        ),
        http.get('/api/v1/projects/{project_id}', ({ response, request }) => {
            const project = projects.find(({ id }) => request.url.endsWith(id));

            if (project !== undefined) {
                return response(200).json(project);
            }

            // @ts-expect-error Issue in OpenApi types
            return response(404).json({
                detail: 'Project not found',
            });
        }),

        http.post('/api/v1/projects', async ({ response, request }) => {
            const body = await request.json();

            projects.push(body as ProjectType);

            // @ts-expect-error We don't rely on the update response in the UI
            return response(201).json(body);
        }),

        http.put('/api/v1/projects/{project_id}', async ({ request, response }) => {
            const body = await request.json();
            const id = request.url.split('/').at(-1);

            projects = projects.map((project) => {
                if (project.id === id) {
                    return {
                        ...project,
                        ...body,
                    } as ProjectType;
                }

                if (body.active != null) {
                    return {
                        ...project,
                        active: false,
                    };
                }

                return project;
            });

            // @ts-expect-error We don't rely on the update response in the UI
            return response(200).json(body);
        }),

        http.delete('/api/v1/projects/{project_id}', async ({ request, response }) => {
            const id = request.url.split('/').at(-1);

            projects = projects.filter((project) => project.id !== id);

            // @ts-expect-error We don't rely on the update response in the UI
            return response(200).json({});
        })
    );

    return projects;
};

class ProjectPage {
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

test.describe('Projects', () => {
    test.describe('Navigation', () => {
        test("Navigates to the project's details page when the URL contains valid project ID", async ({
            network,
            page,
        }) => {
            const project: ProjectType = {
                id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                name: 'Cool project',
                active: true,
            };

            const projectPage = new ProjectPage(page);

            registerApiProjects({ network, defaultProjects: [project] });

            await projectPage.goto(project.id);

            await expect(projectPage.getSelectedProject(project.name)).toBeVisible();
        });

        test('Shows error page when the URL contains invalid project ID', async ({ network, page }) => {
            const project: ProjectType = {
                id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                name: 'Cool project',
                active: true,
            };

            registerApiProjects({ network, defaultProjects: [project] });

            const projectPage = new ProjectPage(page);

            await projectPage.goto('1');

            await expect(projectPage.getSelectedProject(project.name)).toBeHidden();
            await expect(page.getByText('Project not found')).toBeVisible();

            await page.getByRole('button', { name: 'Go back to home page' }).click();

            await expect(projectPage.getSelectedProject(project.name)).toBeVisible();
        });

        test(
            'Navigates to the welcome page when the URL does not contain project ID and there are ' + 'no projects',
            async ({ network, page }) => {
                registerApiProjects({ network, defaultProjects: [] });

                const projectPage = new ProjectPage(page);

                await page.goto(paths.root({}));

                await expect(projectPage.welcomeHeader).toBeVisible();

                await projectPage.create();

                await expect(projectPage.getSelectedProject('Project #1')).toBeVisible();
            }
        );

        test(
            'Navigates to the projects list page when the URL does not contain project ID and there are ' +
                'at least two projects, none of them are active',
            async ({ page, network }) => {
                const projects: ProjectType[] = [
                    {
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: false,
                    },
                    {
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    },
                ];

                registerApiProjects({ network, defaultProjects: projects });
                const projectPage = new ProjectPage(page);

                await page.goto(paths.root({}));

                await expect(projectPage.projectsHeader).toBeVisible();

                for (const project of projects) {
                    await expect(projectPage.getProjectInTheList(project.name)).toBeVisible();
                }

                await projectPage.getProjectInTheList(projects[0].name).click();

                await expect(projectPage.getSelectedProject(projects[0].name)).toBeVisible();
                expect(page.url()).toContain(paths.project({ projectId: projects[0].id }));
            }
        );

        test(
            'Navigates to the project details page of the active project when the URL does not contain project ' +
                'ID and there are at least two projects, one of them is active',
            async ({ page, network }) => {
                const projects: ProjectType[] = [
                    {
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    },
                    {
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    },
                ];

                registerApiProjects({ network, defaultProjects: projects });
                const projectPage = new ProjectPage(page);

                await page.goto(paths.root({}));

                await expect(projectPage.getSelectedProject(projects[0].name)).toBeVisible();
                expect(page.url()).toContain(paths.project({ projectId: projects[0].id }));
            }
        );

        test(
            "Navigates to the project's details page when the URL does not contain project ID and there is " +
                'only one project',
            async ({ network, page }) => {
                const projects: ProjectType[] = [
                    {
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    },
                ];

                registerApiProjects({ network, defaultProjects: projects });

                const projectPage = new ProjectPage(page);

                await page.goto(paths.root({}));

                await expect(projectPage.getSelectedProject(projects[0].name)).toBeVisible();
                expect(page.url()).toContain(paths.project({ projectId: projects[0].id }));
            }
        );

        test('Navigates to projects page when trying to open welcome page and there is at least one project', async ({
            network,
            page,
        }) => {
            const projects: ProjectType[] = [
                {
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: true,
                },
            ];

            registerApiProjects({ network, defaultProjects: projects });

            const projectPage = new ProjectPage(page);

            await page.goto(paths.welcome({}));

            await expect(projectPage.welcomeHeader).toBeHidden();
            await expect(projectPage.getSelectedProject(projects[0].name)).toBeVisible();

            expect(page.url()).toContain(paths.project({ projectId: projects[0].id }));
        });
    });

    test.describe('Project management', () => {
        test('Creates a new project via the project list page with active project', async ({ network, page }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    {
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    },
                    {
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    },
                ],
            });

            const projectPage = new ProjectPage(page);

            await projectPage.gotoProjects();

            await projectPage.create();

            await expect(projectPage.createProjectConfirmationDialogHeading).toBeVisible();

            await projectPage.createConfirmation();

            await expect(projectPage.getSelectedProject('Project #1')).toBeVisible();

            expect(page.url()).toContain(paths.project({ projectId: projects[projects.length - 1].id }));
        });

        test('Creates a new project via the project list page with no active project', async ({ network, page }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    {
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: false,
                    },
                    {
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    },
                ],
            });

            const projectPage = new ProjectPage(page);

            await projectPage.gotoProjects();

            await projectPage.create();

            await expect(projectPage.getSelectedProject('Project #1')).toBeVisible();

            expect(page.url()).toContain(paths.project({ projectId: projects[projects.length - 1].id }));
        });

        test('Creates a new project via the project details page with active project', async ({ network, page }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    {
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    },
                ],
            });

            const projectPage = new ProjectPage(page);

            await projectPage.goto(projects[0].id);

            await projectPage.openProjectManagementPanel();

            await expect(page.getByRole('listitem')).toHaveCount(projects.length);

            await projectPage.create();

            await expect(projectPage.createProjectConfirmationDialogHeading).toBeVisible();

            await projectPage.createConfirmation();

            await expect(projectPage.getSelectedProject('Project #1')).toBeVisible();
            await projectPage.openProjectManagementPanel();
            await expect(page.getByRole('listitem')).toHaveCount(projects.length);

            expect(page.url()).toContain(paths.project({ projectId: projects[projects.length - 1].id }));
        });

        test('Creates a new project via the project details page with no active project', async ({ network, page }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    {
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: false,
                    },
                ],
            });

            const projectPage = new ProjectPage(page);

            await projectPage.goto(projects[0].id);

            await projectPage.openProjectManagementPanel();

            await expect(page.getByRole('listitem')).toHaveCount(projects.length);

            await projectPage.create();

            await expect(projectPage.getSelectedProject('Project #1')).toBeVisible();
            await projectPage.openProjectManagementPanel();
            await expect(page.getByRole('listitem')).toHaveCount(projects.length);

            expect(page.url()).toContain(paths.project({ projectId: projects[projects.length - 1].id }));
        });
    });

    test('Edits a project via the project list page', async ({ network, page }) => {
        const projects = registerApiProjects({
            network,
            defaultProjects: [
                {
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: true,
                },
                {
                    id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #2',
                    active: false,
                },
            ],
        });

        const projectPage = new ProjectPage(page);

        await projectPage.gotoProjects();

        const [project] = projects;
        const newProjectName = 'New Project';

        await projectPage.openProjectMenu(project.name);
        await projectPage.selectMenuItem('Rename');
        await projectPage.updateProjectName(newProjectName);

        await expect(projectPage.getProjectInTheList(project.name)).toBeHidden();
        await expect(projectPage.getProjectInTheList(newProjectName)).toBeVisible();
    });

    test('Edits a project via the project details page', async ({ network, page }) => {
        const projects = registerApiProjects({
            network,
            defaultProjects: [
                {
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: true,
                },
                {
                    id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #2',
                    active: false,
                },
            ],
        });

        const [project] = projects;

        const projectPage = new ProjectPage(page);

        await projectPage.goto(project.id);

        const newProjectName = 'New Project';

        await projectPage.openProjectManagementPanel();
        await projectPage.openProjectMenu(project.name);
        await projectPage.selectMenuItem('Rename');
        await projectPage.updateProjectName(newProjectName);

        await expect(projectPage.getProjectInTheList(project.name)).toBeHidden();
        await expect(projectPage.getProjectInTheList(newProjectName)).toBeVisible();
    });

    test('Deletes a project via the project list page', async ({ network, page }) => {
        const projects = registerApiProjects({
            network,
            defaultProjects: [
                {
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: true,
                },
                {
                    id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #2',
                    active: false,
                },
            ],
        });

        const projectPage = new ProjectPage(page);

        await projectPage.gotoProjects();

        const [project] = projects;

        await projectPage.openProjectMenu(project.name);
        await projectPage.selectMenuItem('Delete');

        await projectPage.delete();

        await expect(projectPage.getProjectInTheList(project.name)).toBeHidden();
    });

    test('Deletes a current project via the project details page', async ({ network, page }) => {
        const projects = registerApiProjects({
            network,
            defaultProjects: [
                {
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: true,
                },
                {
                    id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #2',
                    active: false,
                },
            ],
        });

        const projectPage = new ProjectPage(page);
        const [project] = projects;

        await projectPage.goto(project.id);

        await projectPage.openProjectManagementPanel();
        await projectPage.openProjectMenu(project.name);
        await projectPage.selectMenuItem('Delete');

        await projectPage.delete();

        await expect(projectPage.getProjectInTheList(project.name)).toBeHidden();

        await expect(projectPage.projectsHeader).toBeVisible();
        await expect(page).toHaveURL(paths.projects({}));
    });

    test('Deletes a not current project via the project details page', async ({ network, page }) => {
        const projects = registerApiProjects({
            network,
            defaultProjects: [
                {
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: true,
                },
                {
                    id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #2',
                    active: false,
                },
            ],
        });

        const projectPage = new ProjectPage(page);
        const [project, secondProject] = projects;

        await projectPage.goto(project.id);

        await projectPage.openProjectManagementPanel();
        await projectPage.openProjectMenu(secondProject.name);
        await projectPage.selectMenuItem('Delete');

        await projectPage.delete();

        await expect(projectPage.getProjectInTheList(secondProject.name)).toBeHidden();

        await expect(projectPage.projectsHeader).toBeHidden();

        expect(page.url()).toContain(paths.project({ projectId: project.id }));
    });

    test('Deletes a project via the project details page (the last project)', async ({ network, page }) => {
        const projects = registerApiProjects({
            network,
            defaultProjects: [
                {
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: true,
                },
            ],
        });

        const projectPage = new ProjectPage(page);

        await projectPage.goto(projects[0].id);

        const [project] = projects;

        await projectPage.openProjectManagementPanel();
        await projectPage.openProjectMenu(project.name);
        await projectPage.selectMenuItem('Delete');

        await projectPage.delete();

        await expect(projectPage.getProjectInTheList(project.name)).toBeHidden();

        await expect(projectPage.welcomeHeader).toBeVisible();
        await expect(page).toHaveURL(paths.welcome({}));
    });

    test(
        'Activates an inactive project when there is already an active project using a project menu in the ' +
            'details page',
        async ({ network, page }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    {
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    },
                    {
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    },
                ],
            });

            const projectPage = new ProjectPage(page);

            const [firstProject, secondProject] = projects;

            await projectPage.goto(firstProject.id);

            await expect(projectPage.activeStatus).toBeVisible();
            await projectPage.openProjectManagementPanel();
            await projectPage.openProjectMenu(secondProject.name);
            await projectPage.selectMenuItem('Activate');

            await expect(projectPage.activateProjectDialogHeading).toBeVisible();
            await projectPage.activate();

            await expect(projectPage.inactiveStatus).toBeVisible();

            await projectPage.goto(secondProject.id);
            await expect(projectPage.activeStatus).toBeVisible();
        }
    );

    test(
        'Activates an inactive project when there is no active project using a project menu in the details ' + 'page',
        async ({ network, page }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    {
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: false,
                    },
                    {
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    },
                ],
            });

            const projectPage = new ProjectPage(page);

            const [firstProject, secondProject] = projects;

            await projectPage.goto(firstProject.id);

            await expect(projectPage.activeStatus).toBeVisible();
            await projectPage.openProjectManagementPanel();
            await projectPage.openProjectMenu(secondProject.name);
            await projectPage.selectMenuItem('Activate');

            await expect(projectPage.activateProjectDialogHeading).toBeHidden();

            await expect(projectPage.inactiveStatus).toBeVisible();

            await projectPage.goto(secondProject.id);
            await expect(projectPage.activeStatus).toBeVisible();
        }
    );

    test(
        'Activates an inactive project when there is no active project using the activate button in the ' +
            'current project',
        async ({ network, page }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    {
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: false,
                    },
                    {
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: true,
                    },
                ],
            });

            const projectPage = new ProjectPage(page);

            const [firstProject, secondProject] = projects;

            await projectPage.goto(firstProject.id);

            await expect(projectPage.inactiveStatus).toBeVisible();
            await projectPage.openProjectManagementPanel();

            await projectPage.activateCurrentProject();

            await expect(projectPage.activateProjectDialogHeading).toBeVisible();

            await projectPage.activate();

            await expect(projectPage.activeStatus).toBeVisible();

            await projectPage.goto(secondProject.id);
            await expect(projectPage.inactiveStatus).toBeVisible();
        }
    );

    test('Activates an inactive project in the project list', async ({ network, page }) => {
        const projects = registerApiProjects({
            network,
            defaultProjects: [
                {
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: false,
                },
                {
                    id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #2',
                    active: false,
                },
            ],
        });

        const projectPage = new ProjectPage(page);

        const [firstProject, secondProject] = projects;

        await projectPage.gotoProjects();

        await expect(projectPage.projectsHeader).toBeVisible();

        await expect(projectPage.getInactiveProjectInTheList(firstProject.name)).toBeVisible();
        await expect(projectPage.getInactiveProjectInTheList(secondProject.name)).toBeVisible();

        await projectPage.openProjectMenu(firstProject.name);
        await projectPage.selectMenuItem('Activate');

        await expect(projectPage.activateProjectDialogHeading).toBeHidden();
        await expect(projectPage.getActiveProjectInTheList(firstProject.name)).toBeVisible();
        await expect(projectPage.getInactiveProjectInTheList(secondProject.name)).toBeVisible();
    });

    test('Activates an inactive project using activate in the main content', async ({ network, page }) => {
        const projects = registerApiProjects({
            network,
            defaultProjects: [
                {
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: true,
                },
                {
                    id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #2',
                    active: false,
                },
            ],
        });

        const projectPage = new ProjectPage(page);
        const [firstProject, secondProject] = projects;

        await projectPage.goto(secondProject.id);

        await expect(projectPage.inactiveStatus).toBeVisible();
        await expect(page.getByText('Would you like to activate this project?')).toBeVisible();

        await projectPage.activateCurrentProject();

        await expect(projectPage.activateProjectDialogHeading).toBeVisible();

        await projectPage.activate();

        await expect(projectPage.activeStatus).toBeVisible();

        await projectPage.goto(firstProject.id);
        await expect(projectPage.inactiveStatus).toBeVisible();
    });

    test('Deactivates an active project using a project menu in the details page', async ({ network, page }) => {
        const projects = registerApiProjects({
            network,
            defaultProjects: [
                {
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: true,
                },
                {
                    id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #2',
                    active: false,
                },
            ],
        });

        const projectPage = new ProjectPage(page);

        const [firstProject] = projects;

        await projectPage.goto(firstProject.id);

        await expect(projectPage.activeStatus).toBeVisible();
        await projectPage.openProjectManagementPanel();
        await projectPage.openProjectMenu(firstProject.name);
        await projectPage.selectMenuItem('Deactivate');

        await expect(projectPage.inactiveStatus).toBeVisible();
    });

    test(
        'Deactivates an active project in the details page using the deactivate button in the current' + ' project',
        async ({ network, page }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    {
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    },
                    {
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    },
                ],
            });

            const projectPage = new ProjectPage(page);

            const [firstProject] = projects;

            await projectPage.goto(firstProject.id);

            await expect(projectPage.activeStatus).toBeVisible();
            await projectPage.openProjectManagementPanel();

            await projectPage.deactivateCurrentProject();

            await expect(projectPage.inactiveStatus).toBeVisible();
        }
    );

    test('Deactivates an active project in the project list', async ({ network, page }) => {
        const projects = registerApiProjects({
            network,
            defaultProjects: [
                {
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: true,
                },
                {
                    id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #2',
                    active: false,
                },
            ],
        });

        const projectPage = new ProjectPage(page);

        const [firstProject, secondProject] = projects;

        await projectPage.gotoProjects();

        await expect(projectPage.projectsHeader).toBeVisible();

        await expect(projectPage.getActiveProjectInTheList(firstProject.name)).toBeVisible();
        await expect(projectPage.getInactiveProjectInTheList(secondProject.name)).toBeVisible();

        await projectPage.openProjectMenu(firstProject.name);
        await projectPage.selectMenuItem('Deactivate');

        await expect(projectPage.getInactiveProjectInTheList(firstProject.name)).toBeVisible();
        await expect(projectPage.getInactiveProjectInTheList(secondProject.name)).toBeVisible();
    });
});
