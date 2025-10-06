/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, MouseEventHandler, useState } from 'react';

import { $api } from '@geti-prompt/api';
import { ActionButton, Flex, Grid, Heading, PhotoPlaceholder, repeat, Text, View } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { useNavigate } from 'react-router';

import { SchemaProjectListItem as Project } from '../../../api/openapi-spec';
import { paths } from '../../../routes/paths';
import { useCreateProject } from '../hooks/use-create-project.hook';
import { useDeleteProject } from '../hooks/use-delete-project.hook';
import { useUpdateProject } from '../hooks/use-update-project.hook';
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
    project: Project;
}

const ProjectCard = ({ project }: ProjectCardProps) => {
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
    const [projectIDInEdition, setProjectIdInEdition] = useState<string | null>(null);
    const updateProjectName = useUpdateProject();
    const deleteProject = useDeleteProject();
    const navigate = useNavigate();

    const handleNavigateToProject = () => {
        navigate(paths.project({ projectId: project.id }));
    };

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

        updateProjectName(project.id, newName);
    };

    const handleDelete = () => {
        deleteProject(project.id);
    };

    const isInEditionState = projectIDInEdition === project.id;

    const handleResetProjectInEdition = () => {
        setProjectIdInEdition(null);
    };

    const handleCardClick: MouseEventHandler<HTMLDivElement> = () => {
        if (isInEditionState) {
            handleResetProjectInEdition();

            return;
        }
        handleNavigateToProject();
    };

    return (
        <div
            className={styles.projectCard}
            onClick={handleCardClick}
            aria-label={`Project ${project.name}`}
            role={'listitem'}
        >
            <PhotoPlaceholder name={project.name} indicator={project.id} width={'size-800'} height={'size-800'} />
            <View flex={1} paddingStart={'size-200'} paddingEnd={'size-100'}>
                <Flex justifyContent={'space-between'} alignItems={'center'}>
                    <Heading UNSAFE_className={styles.projectCardTitle}>
                        {isInEditionState ? (
                            <ProjectEdition
                                onBlur={handleBlur}
                                onResetProjectInEdition={handleResetProjectInEdition}
                                name={project.name}
                            />
                        ) : (
                            project.name
                        )}
                    </Heading>

                    <ProjectActions onAction={handleAction} />

                    <DeleteProjectDialog
                        isOpen={isDeleteDialogOpen}
                        onDismiss={() => setIsDeleteDialogOpen(false)}
                        onDelete={handleDelete}
                        projectName={project.name}
                    />
                </Flex>
            </View>
        </div>
    );
};

export const ProjectsListEntry = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');

    const projectsNames = data.projects.map((project) => project.name);

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
                            <ProjectCard project={project} key={project.id} />
                        ))}
                    </Grid>
                </Flex>
            </View>
        </Layout>
    );
};
