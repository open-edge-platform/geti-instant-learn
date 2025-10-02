/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { ActionButton, Flex, Grid, Heading, PhotoPlaceholder, repeat, Text, View } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { useNavigate } from 'react-router';
import { Link } from 'react-router-dom';
import { v4 as uuid } from 'uuid';

import { SchemaProjectListItem as Project } from '../../../api/openapi-spec';
import { paths } from '../../../routes/paths';
import { Layout } from './layout.component';

import styles from './projects-list-entry.module.scss';

interface NewProjectCardProps {
    projectsCount: number;
}

const NewProjectCard = ({ projectsCount }: NewProjectCardProps) => {
    const createProjectMutation = $api.useMutation('post', '/api/v1/projects');
    const navigate = useNavigate();

    const createProject = () => {
        const projectId = uuid();

        createProjectMutation.mutate({
            body: {
                id: projectId,
                name: `Project #${projectsCount + 1}`,
            },
        });

        navigate(paths.project({ projectId }));
    };

    return (
        <View UNSAFE_className={styles.newProjectCard} width={'100%'} height={'100%'}>
            <Flex width={'100%'} height={'100%'} alignItems={'center'}>
                <ActionButton width={'100%'} height={'100%'} onPress={createProject}>
                    <Flex gap={'size-50'} alignItems={'center'}>
                        <AddCircle />
                        <Text>Add Project</Text>
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
    return (
        <Link className={styles.projectCard} to={paths.project({ projectId: project.id })}>
            <PhotoPlaceholder name={project.name} email={project.id} />
            <View flex={1}>
                <Flex justifyContent={'space-between'}>
                    <Heading>{project.name}</Heading>
                </Flex>
            </View>
        </Link>
    );
};

export const ProjectsListEntry = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');

    return (
        <Layout>
            <View maxWidth={'70vw'} minWidth={'50rem'} marginX={'auto'}>
                <Heading level={1} UNSAFE_className={styles.header}>
                    Projects
                </Heading>

                <Grid columns={repeat(2, '1fr')} gap={'size-300'}>
                    <NewProjectCard projectsCount={data.projects.length} />
                    {data.projects.map((project) => (
                        <ProjectCard project={project} key={project.id} />
                    ))}
                </Grid>
            </View>
        </Layout>
    );
};
