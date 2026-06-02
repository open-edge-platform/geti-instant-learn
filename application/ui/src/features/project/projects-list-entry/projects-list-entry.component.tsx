/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { MouseEventHandler } from 'react';

import { $api, type ProjectType } from '@/api';
import { ActionButton, dimensionValue, Flex, Grid, Heading, PhotoPlaceholder, repeat, Text, View } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { Link } from 'react-router-dom';

import { paths } from '../../../constants/paths';
import { useActivateProject } from '../api/use-activate-project.hook';
import { useCreateProject } from '../api/use-create-project.hook';
import { ProjectActions } from '../project-list-item/project-actions.component';
import { generateUniqueProjectName } from '../utils';
import { Layout } from './layout.component';

import classes from './projects-list-entry.module.scss';

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
        <View UNSAFE_className={classes.newProjectCard} width={'100%'} height={'100%'}>
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
    const activateProject = useActivateProject();

    const handleCardClick: MouseEventHandler<HTMLAnchorElement> = () => {
        if (project.active) {
            return;
        }

        activateProject.mutate(project, activeProject);
    };

    return (
        <View position={'relative'}>
            <Link
                data-active={project.active}
                to={paths.project({ projectId: project.id })}
                className={clsx(classes.projectCard, classes.projectCardHovered)}
                onClick={handleCardClick}
                role={'listitem'}
                aria-label={`Project ${project.name}`}
            >
                <PhotoPlaceholder name={project.name} indicator={project.id} width={'size-800'} height={'size-800'} />
                <View flex={1} paddingStart={'size-200'} paddingEnd={'size-500'}>
                    <Flex justifyContent={'space-between'} alignItems={'center'} height={'100%'}>
                        <Heading UNSAFE_className={classes.projectCardTitle} margin={0}>
                            {project.name}
                        </Heading>
                    </Flex>
                </View>
            </Link>

            <ProjectActions
                projectId={project.id}
                projectName={project.name}
                projectNames={projectNames}
                actionButtonStyle={{
                    position: 'absolute',
                    right: dimensionValue('size-200'),
                    top: '50%',
                    transform: 'translateY(-50%)',
                }}
            />
        </View>
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
                    <Heading level={1} UNSAFE_className={classes.header} marginBottom={'size-100'}>
                        Projects
                    </Heading>
                    <Text UNSAFE_className={classes.description} marginBottom={'size-500'}>
                        Create projects to keep each task focused, with its own prompts, examples, and results.
                    </Text>

                    <Grid
                        columns={repeat(2, 'minmax(0, 1fr)')}
                        gap={'size-300'}
                        flex={1}
                        alignContent={'start'}
                        UNSAFE_className={classes.projectsList}
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
