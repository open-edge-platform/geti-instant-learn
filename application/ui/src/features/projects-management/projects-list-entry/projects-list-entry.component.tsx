/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, MouseEventHandler, useState } from 'react';

import { $api, type ProjectType } from '@geti-prompt/api';
import { ActionButton, Flex, Grid, Heading, PhotoPlaceholder, repeat, Text, View } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { Link } from 'react-router-dom';

import { paths } from '../../../routes/paths';
import { ActivateProjectDialog } from '../activate-project-dialog/activate-project-dialog.component';
import { useCreateProject } from '../hooks/use-create-project.hook';
import { useDeleteProject } from '../hooks/use-delete-project.hook';
import { useProjectActivityManagement } from '../hooks/use-project-activity-management.hook';
import { useUpdateProject } from '../hooks/use-update-project.hook';
import { ProjectActivityStatus } from '../project-activity-status/project-activity-status.component';
import {
    DeleteProjectDialog,
    PROJECT_ACTIONS,
    ProjectActions,
    ProjectEdition,
} from '../project-list-item/project-actions.component';
import { generateUniqueProjectName } from '../utils';
import { Layout } from './layout.component';

import styles from './projects-list-entry.module.scss';

interface NewProjectCardProps {
    projectsNames: string[];
}

const NewProjectCard = ({ projectsNames }: NewProjectCardProps) => {
    const createProject = useCreateProject();

    const handleCreateProject = () => {
        const projectName = generateUniqueProjectName(projectsNames);
        createProject(projectName);
    };

    return (
        <View UNSAFE_className={styles.newProjectCard} width={'100%'} height={'100%'}>
            <Flex width={'100%'} height={'100%'} alignItems={'center'}>
                <ActionButton width={'100%'} height={'100%'} onPress={handleCreateProject}>
                    <Flex gap={'size-50'} alignItems={'center'}>
                        <AddCircle />
                        <Text>Create project</Text>
                    </Flex>
                </ActionButton>
            </Flex>
        </View>
    );
};

interface ProjectCardProps {
    project: ProjectType;
    activedProject: ProjectType | undefined;
    projectNames: string[];
}

const ProjectCard = ({ project, activedProject, projectNames }: ProjectCardProps) => {
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
    const [projectIDInEdition, setProjectIdInEdition] = useState<string | null>(null);
    const updateProjectName = useUpdateProject();
    const deleteProject = useDeleteProject();
    const { isVisible, onClose, onShowActivateProjectDialog, onActivate, onDeactivate } = useProjectActivityManagement(
        project.id
    );

    const handleAction = (key: Key) => {
        if (key === PROJECT_ACTIONS.RENAME) {
            setProjectIdInEdition(project.id);
        } else if (key === PROJECT_ACTIONS.DELETE) {
            setIsDeleteDialogOpen(true);
        } else if (key === PROJECT_ACTIONS.ACTIVATE) {
            if (activedProject === undefined) {
                onActivate();
                return;
            }

            onShowActivateProjectDialog();
        } else if (key === PROJECT_ACTIONS.DEACTIVATE) {
            onDeactivate();
        }
    };

    const handleBlur = (newName: string) => {
        if (newName === project.name) return;
        if (newName.trim().length === 0) return;

        updateProjectName(project.id, { name: newName });
    };

    const handleDelete = () => {
        deleteProject(project.id);
    };

    const isInEditionState = projectIDInEdition === project.id;

    const handleResetProjectInEdition = () => {
        setProjectIdInEdition(null);
    };

    const handleCardClick: MouseEventHandler<HTMLAnchorElement> = (event) => {
        if (isInEditionState) {
            event.preventDefault();

            return;
        }
    };

    const actions = [
        PROJECT_ACTIONS.RENAME,
        PROJECT_ACTIONS.DELETE,
        project.active ? PROJECT_ACTIONS.DEACTIVATE : PROJECT_ACTIONS.ACTIVATE,
    ];

    return (
        <Link
            to={paths.project({ projectId: project.id })}
            className={styles.projectCard}
            onClick={handleCardClick}
            role={'listitem'}
            aria-label={`Project ${project.name}`}
        >
            <PhotoPlaceholder name={project.name} indicator={project.id} width={'size-800'} height={'size-800'} />
            <View flex={1} paddingStart={'size-200'} paddingEnd={'size-100'}>
                <Flex justifyContent={'space-between'} alignItems={'center'}>
                    <Heading UNSAFE_className={styles.projectCardTitle}>
                        {isInEditionState ? (
                            <ProjectEdition
                                projectNames={projectNames}
                                onBlur={handleBlur}
                                onResetProjectInEdition={handleResetProjectInEdition}
                                name={project.name}
                            />
                        ) : (
                            project.name
                        )}
                        <ProjectActivityStatus isActive={project.active} />
                    </Heading>

                    <ProjectActions actions={actions} onAction={handleAction} />

                    <DeleteProjectDialog
                        isOpen={isDeleteDialogOpen}
                        onDismiss={() => setIsDeleteDialogOpen(false)}
                        onDelete={handleDelete}
                        projectName={project.name}
                    />
                    {/*
                        Activate Project Dialog is only visible when there is already an active project.
                        When there is no active project, the dialog is not visible; we just activate the selected
                        project directly.
                    */}
                    <ActivateProjectDialog
                        isVisible={isVisible}
                        onClose={onClose}
                        activeProjectName={activedProject?.name ?? ''}
                        inactiveProjectName={project.name}
                        onActivate={onActivate}
                    />
                </Flex>
            </View>
        </Link>
    );
};

export const ProjectsListEntry = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');

    const projectsNames = data.projects.map((project) => project.name);
    const activeProject = data.projects.find((project) => project.active);

    return (
        <Layout>
            <View maxWidth={'70vw'} minWidth={'50rem'} marginX={'auto'} height={'100%'}>
                <Flex direction={'column'} height={'100%'}>
                    <Heading level={1} UNSAFE_className={styles.header} marginBottom={'size-500'}>
                        Projects
                    </Heading>

                    <Grid
                        columns={repeat(2, '1fr')}
                        gap={'size-300'}
                        flex={1}
                        alignContent={'start'}
                        UNSAFE_className={styles.projectsList}
                    >
                        <NewProjectCard projectsNames={projectsNames} />
                        {data.projects.map((project) => (
                            <ProjectCard
                                project={project}
                                key={project.id}
                                projectNames={projectsNames.filter((name) => name !== project.name)}
                                activedProject={activeProject}
                            />
                        ))}
                    </Grid>
                </Flex>
            </View>
        </Layout>
    );
};
