/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { isEmpty } from 'lodash-es';

import { Project, ProjectListItem } from './project-list-item/project-list-item.component';

import styles from './projects-list.module.scss';

interface ProjectListProps {
    projects: Project[];
    projectIdInEdition: string | null;
    setProjectInEdition: (projectId: string | null) => void;
}

export const ProjectsList = ({ projects, setProjectInEdition, projectIdInEdition }: ProjectListProps) => {
    const updateProjectMutation = $api.useMutation('put', '/api/v1/projects/{project_id}');
    const deleteProjectMutation = $api.useMutation('delete', '/api/v1/projects/{project_id}');

    const updateProjectName = (id: string, name: string): void => {
        updateProjectMutation.mutate({
            body: {
                name,
            },
            params: {
                path: {
                    project_id: id,
                },
            },
        });
    };

    const deleteProject = (id: string): void => {
        deleteProjectMutation.mutate({
            params: {
                path: {
                    project_id: id,
                },
            },
        });
    };

    const isInEditionMode = (projectId: string) => {
        return projectIdInEdition === projectId;
    };

    const handleBlur = (projectId: string, newName: string) => {
        setProjectInEdition(null);

        const projectToUpdate = projects.find((project) => project.id === projectId);
        if (projectToUpdate?.name === newName || isEmpty(newName.trim())) {
            return;
        }

        updateProjectName(projectId, newName);
    };

    const handleRename = (projectId: string) => {
        setProjectInEdition(projectId);
    };

    return (
        <ul className={styles.projectList}>
            {projects.map((project) => (
                <ProjectListItem
                    key={project.id}
                    project={project}
                    onRename={handleRename}
                    onDelete={deleteProject}
                    onBlur={handleBlur}
                    isInEditMode={isInEditionMode(project.id)}
                />
            ))}
        </ul>
    );
};
