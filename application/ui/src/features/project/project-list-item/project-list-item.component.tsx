/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { MouseEventHandler } from 'react';

import { type ProjectType } from '@/api';
import { Flex, PhotoPlaceholder, Text } from '@geti/ui';
import { Link } from 'react-router-dom';

import { paths } from '../../../constants/paths';
import { ProjectActions } from './project-actions.component';

import styles from './project-list-item.module.scss';

interface ProjectListItemProps {
    project: ProjectType;
    projectNames: string[];
    onActivateProject: (project: ProjectType) => void;
}

export const ProjectListItem = ({ project, projectNames, onActivateProject }: ProjectListItemProps) => {
    const handleItemClick: MouseEventHandler<HTMLAnchorElement> = () => {
        onActivateProject(project);
    };

    return (
        <li className={styles.projectListItem} aria-label={`Project ${project.name}`} data-active={project.active}>
            <Link to={paths.project({ projectId: project.id })} onClick={handleItemClick}>
                <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                    <Flex alignItems={'center'} gap={'size-100'}>
                        <PhotoPlaceholder
                            name={project.name}
                            indicator={project.id}
                            height={'size-325'}
                            width={'size-325'}
                        />
                        <Text>{project.name}</Text>
                    </Flex>
                    <ProjectActions projectId={project.id} projectName={project.name} projectNames={projectNames} />
                </Flex>
            </Link>
        </li>
    );
};
