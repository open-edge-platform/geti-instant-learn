/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, KeyboardEvent, useEffect, useRef, useState } from 'react';

import { ActionMenu, AlertDialog, DialogContainer, Item, TextField, TextFieldRef } from '@geti/ui';

import styles from './project-list-item.module.scss';

interface ProjectEditionProps {
    onBlur: (newName: string) => void;
    name: string;
}

export const ProjectEdition = ({ name, onBlur }: ProjectEditionProps) => {
    const textFieldRef = useRef<TextFieldRef>(null);
    const [newName, setNewName] = useState<string>(name);

    const handleBlur = () => {
        onBlur(newName);
    };

    const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            onBlur(newName);
        } else if (e.key === 'Escape') {
            e.preventDefault();
            setNewName(name);
            onBlur(name);
        }
    };

    useEffect(() => {
        textFieldRef.current?.select();
    }, []);

    return (
        <TextField
            isQuiet
            ref={textFieldRef}
            value={newName}
            onBlur={handleBlur}
            onKeyDown={handleKeyDown}
            onChange={setNewName}
            aria-label='Edit project name'
        />
    );
};

export const PROJECT_ACTIONS = {
    RENAME: 'Rename',
    DELETE: 'Delete',
};

interface ProjectActionsProps {
    onAction: (key: Key) => void;
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

export const ProjectActions = ({ onAction }: ProjectActionsProps) => {
    return (
        <ActionMenu isQuiet onAction={onAction} aria-label={'Project actions'} UNSAFE_className={styles.actionMenu}>
            {[PROJECT_ACTIONS.RENAME, PROJECT_ACTIONS.DELETE].map((action) => (
                <Item key={action}>{action}</Item>
            ))}
        </ActionMenu>
    );
};
