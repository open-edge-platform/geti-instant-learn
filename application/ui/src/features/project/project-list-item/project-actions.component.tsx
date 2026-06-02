/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CSSProperties, FormEvent, Key, useState } from 'react';

import { useDeleteProject } from '@/features/project/api/use-delete-project.hook';
import { useUpdateProject } from '@/features/project/api/use-update-project.hook';
import {
    ActionButton,
    AlertDialog,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Form,
    Heading,
    Item,
    Menu,
    MenuTrigger,
    TextField,
    toast,
} from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

export const PROJECT_ACTIONS = {
    RENAME: 'Rename',
    DELETE: 'Delete',
};

type EditProjectNameDialogProps = {
    onClose: () => void;
    isOpen: boolean;
    projectId: string;
    projectName: string;
    projectNames: string[];
};

const PROJECT_NAME_MAX_LENGTH = 100;

export const EditProjectNameDialog = ({
    onClose,
    isOpen,
    projectId,
    projectName,
    projectNames,
}: EditProjectNameDialogProps) => {
    const [newProjectName, setNewProjectName] = useState(projectName);
    const updateProject = useUpdateProject();

    const trimmedProjectName = newProjectName.trim();
    const isNameUnchanged = trimmedProjectName === projectName;

    const validateProjectName = (inputName: string) => {
        if (inputName.trim() === '') {
            return 'Project name cannot be empty';
        }

        if (projectNames.includes(inputName.trim())) {
            return 'That project name already exists';
        }

        return undefined;
    };

    const validationErrorMessage = validateProjectName(newProjectName);
    const isSaveButtonDisabled =
        isEmpty(trimmedProjectName) ||
        isNameUnchanged ||
        updateProject.isPending ||
        validationErrorMessage !== undefined;

    const editProjectName = (newName: string) => {
        updateProject.mutate(
            projectId,
            {
                name: newName,
            },
            () => {
                onClose();
                toast({ type: 'success', message: 'Project updated successfully' });
            }
        );
    };

    const handleEditProjectName = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (isSaveButtonDisabled) {
            return;
        }

        editProjectName(newProjectName);
    };

    return (
        <DialogContainer onDismiss={onClose}>
            {isOpen && (
                <Dialog>
                    <Heading>Edit project name</Heading>
                    <Divider />
                    <Content>
                        <Form onSubmit={handleEditProjectName}>
                            <TextField
                                //eslint-disable-next-line jsx-a11y/no-autofocus
                                autoFocus
                                maxLength={PROJECT_NAME_MAX_LENGTH}
                                value={newProjectName}
                                onChange={setNewProjectName}
                                width='100%'
                                aria-label='Edit project name'
                                isReadOnly={updateProject.isPending}
                                errorMessage={validationErrorMessage}
                                validationState={validationErrorMessage === undefined ? undefined : 'invalid'}
                            />
                            <ButtonGroup align={'end'} marginTop={'size-350'}>
                                <Button variant='secondary' onPress={onClose} isDisabled={updateProject.isPending}>
                                    Cancel
                                </Button>
                                <Button
                                    type='submit'
                                    variant='accent'
                                    isDisabled={isSaveButtonDisabled}
                                    isPending={updateProject.isPending}
                                >
                                    Save
                                </Button>
                            </ButtonGroup>
                        </Form>
                    </Content>
                </Dialog>
            )}
        </DialogContainer>
    );
};

interface DeleteProjectDialogProps {
    isOpen: boolean;
    projectId: string;
    onDismiss: () => void;
    onSuccess?: () => void;
    projectName: string;
}

export const DeleteProjectDialog = ({
    isOpen,
    projectId,
    onDismiss,
    projectName,
    onSuccess,
}: DeleteProjectDialogProps) => {
    const deleteProject = useDeleteProject();

    const handleDelete = (): void => {
        deleteProject.mutate(projectId, () => {
            onDismiss();
            onSuccess?.();
        });
    };

    return (
        <DialogContainer onDismiss={onDismiss}>
            {isOpen && (
                <AlertDialog
                    title='Delete'
                    variant='destructive'
                    primaryActionLabel='Delete'
                    onPrimaryAction={handleDelete}
                    isPrimaryActionDisabled={deleteProject.isPending}
                    cancelLabel={'Cancel'}
                >
                    {`Are you sure you want to delete project "${projectName}"?`}
                </AlertDialog>
            )}
        </DialogContainer>
    );
};

const PROJECT_ACTIONS_ITEMS = [PROJECT_ACTIONS.RENAME, PROJECT_ACTIONS.DELETE];

interface ProjectActionsProps {
    projectId: string;
    projectName: string;
    projectNames: string[];
    actionButtonStyle?: CSSProperties;
    onDeleted?: () => void;
}

export const ProjectActions = ({
    projectId,
    actionButtonStyle,
    projectName,
    projectNames,
    onDeleted,
}: ProjectActionsProps) => {
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
    const [isEditDialogOpen, setIsEditDialogOpen] = useState<boolean>(false);

    const handleAction = (key: Key) => {
        if (key === PROJECT_ACTIONS.RENAME) {
            setIsEditDialogOpen(true);
        } else if (key === PROJECT_ACTIONS.DELETE) {
            setIsDeleteDialogOpen(true);
        }
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label={'Project actions'} UNSAFE_style={actionButtonStyle}>
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={handleAction}>
                    {PROJECT_ACTIONS_ITEMS.map((label) => (
                        <Item key={label}>{label}</Item>
                    ))}
                </Menu>
            </MenuTrigger>

            <DeleteProjectDialog
                isOpen={isDeleteDialogOpen}
                projectId={projectId}
                onDismiss={() => setIsDeleteDialogOpen(false)}
                onSuccess={onDeleted}
                projectName={projectName}
            />

            <EditProjectNameDialog
                isOpen={isEditDialogOpen}
                onClose={() => setIsEditDialogOpen(false)}
                projectId={projectId}
                projectName={projectName}
                projectNames={projectNames}
            />
        </>
    );
};
