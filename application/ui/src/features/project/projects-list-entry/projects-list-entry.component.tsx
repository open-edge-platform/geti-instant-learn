/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, MouseEventHandler, useState } from 'react';

import { $api, type ProjectType } from '@/api';
import { ActionButton, Flex, Grid, Heading, PhotoPlaceholder, repeat, Text, View } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { Link } from 'react-router-dom';

import { paths } from '../../../constants/paths';
import { useActivateProject } from '../api/use-activate-project.hook';
import { useCreateProject } from '../api/use-create-project.hook';
import { useDeleteProject } from '../api/use-delete-project.hook';
import { useUpdateProject } from '../api/use-update-project.hook';
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
        const name = generateUniqueProjectName(projectsNames);
        createProject.mutate({ name });
    };

    return (
        <View UNSAFE_className={styles.newProjectCard} width={'100%'} height={'100%'}>
            <Flex width={'100%'} height={'100%'} alignItems={'center'}>
                <ActionButton
                    width={'100%'}
                    height={'100%'}
                    onPress={handleCreateProject}
                    isDisabled={createProject.isPending}
                >
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
    projectNames: string[];
    activeProject: ProjectType | undefined;
}

const ProjectCard = ({ project, projectNames, activeProject }: ProjectCardProps) => {
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
    const [projectIDInEdition, setProjectIdInEdition] = useState<string | null>(null);
    const updateProject = useUpdateProject();
    const deleteProject = useDeleteProject();
    const activateProject = useActivateProject();

    const handleAction = (key: Key) => {
        if (key === PROJECT_ACTIONS.RENAME) {
            setProjectIdInEdition(project.id);
        } else if (key === PROJECT_ACTIONS.DELETE) {
            setIsDeleteDialogOpen(true);
        }
    };

    const handleBlur = (newName: string) => {
        if (newName === project.name) return;
        if (newName.trim().length === 0) return;

        updateProject.mutate(project.id, { name: newName });
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

        if (project.active) {
            return;
        }

        activateProject.mutate(project, activeProject);
    };

    const actions = [PROJECT_ACTIONS.RENAME, PROJECT_ACTIONS.DELETE];

    return (
        <Link
            data-active={project.active}
            to={paths.project({ projectId: project.id })}
            className={clsx(styles.projectCard, {
                [styles.projectCardHovered]: !isInEditionState,
            })}
            onClick={handleCardClick}
            role={'listitem'}
            aria-label={`Project ${project.name}`}
        >
            <PhotoPlaceholder name={project.name} indicator={project.id} width={'size-800'} height={'size-800'} />
            <View flex={1} paddingStart={'size-200'} paddingEnd={'size-100'}>
                <Flex justifyContent={'space-between'} alignItems={'center'} height={'100%'}>
                    <Heading UNSAFE_className={styles.projectCardTitle} margin={0}>
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
                    </Heading>

                    <ProjectActions actions={actions} onAction={handleAction} />

                    <DeleteProjectDialog
                        isOpen={isDeleteDialogOpen}
                        onDismiss={() => setIsDeleteDialogOpen(false)}
                        onDelete={handleDelete}
                        projectName={project.name}
                    />
                </Flex>
            </View>
        </Link>
    );
};

export const ProjectsListEntry = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');

    const projectsNames = data.projects.map((project) => project.name);
    const activeProject = data.projects.find(({ active }) => active);

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
                                key={project.id}
                                project={project}
                                projectNames={projectsNames.filter((name) => name !== project.name)}
                                activeProject={activeProject}
                            />
                        ))}
                    </Grid>
                </Flex>
            </View>
        </Layout>
    );
};
