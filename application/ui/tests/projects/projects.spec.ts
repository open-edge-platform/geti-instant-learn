/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, http, test } from '@geti-prompt/test-fixtures';
import { NetworkFixture } from '@msw/playwright';

import { ProjectType } from '../../src/api';
import { paths } from '../../src/constants/paths';
import { getMockedProject } from '../../src/test-utils/mocks/mock-project';

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
        http.get('/api/v1/projects/{project_id}', ({ response, params }) => {
            const project = projects.find(({ id }) => id === params.project_id);

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

            projects.push({ ...body, active: true } as ProjectType);

            // @ts-expect-error We don't rely on the update response in the UI
            return response(201).json(body);
        }),

        http.put('/api/v1/projects/{project_id}', async ({ request, response, params }) => {
            const body = await request.json();

            const id = params.project_id;

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

            return response(204).empty();
        })
    );

    return projects;
};

test.describe('Projects', () => {
    test.describe('Navigation', () => {
        test("Navigates to the project's details page when the URL contains valid project ID", async ({
            network,
            projectPage,
        }) => {
            const project: ProjectType = getMockedProject({
                id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                name: 'Cool project',
                active: true,
            });

            registerApiProjects({ network, defaultProjects: [project] });

            await projectPage.goto(project.id);

            await expect(projectPage.getSelectedProject(project.name)).toBeVisible();
        });

        test('Shows error page when the URL contains invalid project ID', async ({ network, page, projectPage }) => {
            const project: ProjectType = getMockedProject({
                id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                name: 'Cool project',
                active: true,
            });

            registerApiProjects({ network, defaultProjects: [project] });

            await projectPage.goto('1');

            await expect(projectPage.getSelectedProject(project.name)).toBeHidden();
            await expect(page.getByText('Project not found')).toBeVisible();

            await page.getByRole('button', { name: 'Go back to home page' }).click();

            await expect(projectPage.getSelectedProject(project.name)).toBeVisible();
        });

        test(
            'Navigates to the welcome page when the URL does not contain project ID and there are ' + 'no projects',
            async ({ network, page, projectPage }) => {
                registerApiProjects({ network, defaultProjects: [] });

                await page.goto(paths.root({}));

                await expect(projectPage.welcomeHeader).toBeVisible();

                await projectPage.create();

                await expect(projectPage.getSelectedProject('Project #1')).toBeVisible();
            }
        );

        test(
            'Navigates to the project details page of the active project when the URL does not contain project ' +
                'ID and there are at least two projects, one of them is active',
            async ({ page, network, projectPage }) => {
                const projects: ProjectType[] = [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                    getMockedProject({
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    }),
                ];

                registerApiProjects({ network, defaultProjects: projects });

                await page.goto(paths.root({}));

                await expect(projectPage.getSelectedProject(projects[0].name)).toBeVisible();
                expect(page.url()).toContain(paths.project({ projectId: projects[0].id }));
            }
        );

        test(
            "Navigates to the project's details page when the URL does not contain project ID and there is " +
                'only one project',
            async ({ network, page, projectPage }) => {
                const projects: ProjectType[] = [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                ];

                registerApiProjects({ network, defaultProjects: projects });

                await page.goto(paths.root({}));

                await expect(projectPage.getSelectedProject(projects[0].name)).toBeVisible();
                expect(page.url()).toContain(paths.project({ projectId: projects[0].id }));
            }
        );

        test('Navigates to projects page when trying to open welcome page and there is at least one project', async ({
            network,
            page,
            projectPage,
        }) => {
            const projects: ProjectType[] = [
                getMockedProject({
                    id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                    name: 'Cool project #1',
                    active: true,
                }),
            ];

            registerApiProjects({ network, defaultProjects: projects });

            await page.goto(paths.welcome({}));

            await expect(projectPage.welcomeHeader).toBeHidden();
            await expect(projectPage.getSelectedProject(projects[0].name)).toBeVisible();

            expect(page.url()).toContain(paths.project({ projectId: projects[0].id }));
        });
    });

    test.describe('Project management', () => {
        test('Creates a new project via the project list page', async ({ network, page, projectPage }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                    getMockedProject({
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    }),
                ],
            });

            await projectPage.gotoProjects();

            await projectPage.create();

            const project = projectPage.getSelectedProject('Project #1');

            await expect(project).toBeVisible();
            await expect(project).toHaveAttribute('data-active', 'true');

            await projectPage.openProjectManagementPanel();
            await expect(page.getByRole('listitem')).toHaveCount(projects.length);

            expect(page.url()).toContain(paths.project({ projectId: projects[projects.length - 1].id }));
        });

        test('Creates a new project via the project details page', async ({ network, page, projectPage }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                ],
            });

            await projectPage.goto(projects[0].id);

            await projectPage.openProjectManagementPanel();

            await expect(page.getByRole('listitem')).toHaveCount(projects.length);

            await projectPage.create();

            const project = projectPage.getSelectedProject('Project #1');
            await expect(project).toBeVisible();
            await expect(project).toHaveAttribute('data-active', 'true');

            await projectPage.openProjectManagementPanel();
            await expect(page.getByRole('listitem')).toHaveCount(projects.length);

            expect(page.url()).toContain(paths.project({ projectId: projects[projects.length - 1].id }));
        });

        test('Edits a project via the project list page', async ({ network, projectPage }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                    getMockedProject({
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    }),
                ],
            });

            await projectPage.gotoProjects();

            const [project] = projects;
            const newProjectName = 'New Project';

            await projectPage.openProjectMenu(project.name);
            await projectPage.selectMenuItem('Rename');
            await projectPage.updateProjectName(newProjectName);

            await expect(projectPage.getProjectInTheList(project.name)).toBeHidden();
            await expect(projectPage.getProjectInTheList(newProjectName)).toBeVisible();
        });

        test('Edits a project via the project details page', async ({ network, projectPage }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                    getMockedProject({
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    }),
                ],
            });

            const [project] = projects;

            await projectPage.goto(project.id);

            const newProjectName = 'New Project';

            await projectPage.openProjectManagementPanel();
            await projectPage.openProjectMenu(project.name);
            await projectPage.selectMenuItem('Rename');
            await projectPage.updateProjectName(newProjectName);

            await expect(projectPage.getProjectInTheList(project.name)).toBeHidden();
            await expect(projectPage.getProjectInTheList(newProjectName)).toBeVisible();
        });

        test('Deletes a project via the project list page', async ({ network, projectPage }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                    getMockedProject({
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    }),
                ],
            });

            await projectPage.gotoProjects();

            const [project] = projects;

            await projectPage.openProjectMenu(project.name);
            await projectPage.selectMenuItem('Delete');

            await projectPage.delete();

            await expect(projectPage.getProjectInTheList(project.name)).toBeHidden();
        });

        test('Deletes a current project via the project details page', async ({ network, page, projectPage }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                    getMockedProject({
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    }),
                ],
            });
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

        test('Deletes a not current project via the project details page', async ({ network, page, projectPage }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                    getMockedProject({
                        id: '10f1d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    }),
                ],
            });
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

        test('Deletes a project via the project details page (the last project)', async ({
            network,
            page,
            projectPage,
        }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                ],
            });

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

        test('Activates a project when selecting it from the project details page in project list', async ({
            network,
            projectPage,
        }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                    getMockedProject({
                        id: 'd4231d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    }),
                ],
            });

            const [firstProject, secondProject] = projects;

            await projectPage.goto(firstProject.id);

            await expect(projectPage.getSelectedProject(firstProject.name)).toHaveAttribute('data-active', 'true');

            await projectPage.openProjectManagementPanel();

            await projectPage.openProject(secondProject.name);

            await expect(projectPage.getSelectedProject(secondProject.name)).toHaveAttribute('data-active', 'true');
        });

        test('Activates a project when selecting it from the projects page in project list', async ({
            network,
            projectPage,
        }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                    getMockedProject({
                        id: 'd4231d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    }),
                ],
            });

            const [firstProject, secondProject] = projects;

            await projectPage.gotoProjects();

            await expect(projectPage.getProjectInTheList(firstProject.name)).toHaveAttribute('data-active', 'true');
            await expect(projectPage.getProjectInTheList(secondProject.name)).toHaveAttribute('data-active', 'false');

            await projectPage.openProject(secondProject.name);

            await expect(projectPage.getSelectedProject(secondProject.name)).toHaveAttribute('data-active', 'true');
        });

        test('Activates a project when opening a link with a project that is inactive', async ({
            network,
            projectPage,
        }) => {
            const projects = registerApiProjects({
                network,
                defaultProjects: [
                    getMockedProject({
                        id: '10f1d4b7-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #1',
                        active: true,
                    }),
                    getMockedProject({
                        id: 'd4231d423-4a1e-40ed-b025-2c4811f87c95',
                        name: 'Cool project #2',
                        active: false,
                    }),
                ],
            });

            const [, inactiveProject] = projects;

            await projectPage.goto(inactiveProject.id);

            const project = projectPage.getSelectedProject(inactiveProject.name);

            await expect(project).toBeVisible();

            await expect(projectPage.getSelectedProject(inactiveProject.name)).toHaveAttribute('data-active', 'true');
        });
    });
});
