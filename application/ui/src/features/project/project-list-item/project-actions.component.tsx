/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, KeyboardEvent, useEffect, useRef, useState } from 'react';

import { useOnOutsideClick } from '@geti-prompt/hooks';
import { ActionMenu, AlertDialog, DialogContainer, Item, TextField, TextFieldRef } from '@geti/ui';

import styles from './project-list-item.module.scss';

interface ProjectEditionProps {
    onBlur: (newName: string) => void;
    onResetProjectInEdition: () => void;
    name: string;
    projectNames: string[];
}

export const ProjectEdition = ({ name, onBlur, onResetProjectInEdition, projectNames }: ProjectEditionProps) => {
    const textFieldRef = useRef<TextFieldRef>(null);
    const [newName, setNewName] = useState<string>(name);

    const handleBlur = () => {
        const errorMessage = validateProjectName(newName);

        if (errorMessage !== undefined) {
            return;
        }

        onBlur(newName.trim());
    };

    const validateProjectName = (inputName: string) => {
        if (inputName.trim() === '') {
            return 'Project name cannot be empty';
        }

        if (projectNames.includes(inputName.trim())) {
            return 'That project name already exists';
        }

        return undefined;
    };

    const handleKeyDown = (event: KeyboardEvent) => {
        if (event.key === 'Enter') {
            event.preventDefault();

            const errorMessage = validateProjectName(newName);

            if (errorMessage !== undefined) {
                return;
            }

            handleBlur();
            onResetProjectInEdition();
        } else if (event.key === 'Escape') {
            event.preventDefault();

            setNewName(name);
            onBlur(name);
            onResetProjectInEdition();
        }
    };

    useEffect(() => {
        textFieldRef.current?.select();
    }, []);

    useOnOutsideClick(textFieldRef, onResetProjectInEdition);

    const errorMessage = validateProjectName(newName);

    return (
        <div
            onClick={(event) => {
                // The goal of those two lines is to prevent the click event from bubbling up to the parent
                // and triggering the default behavior of the Link (navigation) when being in edit mode.
                event.stopPropagation();
                event.preventDefault();
            }}
        >
            <TextField
                isQuiet
                ref={textFieldRef}
                value={newName}
                onBlur={handleBlur}
                onKeyDown={handleKeyDown}
                onChange={setNewName}
                aria-label='Edit project name'
                errorMessage={errorMessage}
                validationState={errorMessage ? 'invalid' : undefined}
            />
        </div>
    );
};

export const PROJECT_ACTIONS = {
    RENAME: 'Rename',
    DELETE: 'Delete',
    ACTIVATE: 'Activate',
    DEACTIVATE: 'Deactivate',
};

interface ProjectActionsProps {
    onAction: (key: Key) => void;
    actions: string[];
}

interface DeleteProjectDialogProps {
    isOpen: boolean;
    onDismiss: () => void;
    onDelete: () => void;
    projectName: string;
}

export const DeleteProjectDialog = ({ isOpen, onDismiss, projectName, onDelete }: DeleteProjectDialogProps) => {
    return (
        <DialogContainer onDismiss={onDismiss}>
            {isOpen && (
                <AlertDialog
                    title='Delete'
                    variant='destructive'
                    primaryActionLabel='Delete'
                    onPrimaryAction={onDelete}
                    cancelLabel={'Cancel'}
                >
                    {`Are you sure you want to delete project "${projectName}"?`}
                </AlertDialog>
            )}
        </DialogContainer>
    );
};

export const ProjectActions = ({ onAction, actions }: ProjectActionsProps) => {
    return (
        <ActionMenu isQuiet onAction={onAction} aria-label={'Project actions'} UNSAFE_className={styles.actionMenu}>
            {actions.map((action) => (
                <Item key={action}>{action}</Item>
            ))}
        </ActionMenu>
    );
};
