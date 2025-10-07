/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, KeyboardEvent, RefObject, useEffect, useRef, useState } from 'react';

import { ActionMenu, AlertDialog, DialogContainer, Item, TextField, TextFieldRef } from '@geti/ui';

import styles from './project-list-item.module.scss';

interface ProjectEditionProps {
    onBlur: (newName: string) => void;
    onResetProjectInEdition: () => void;
    name: string;
}

const useOnOutsideClick = (textFieldRef: RefObject<TextFieldRef | null>, onClickOutside: () => void) => {
    const resetProjectInEditionRef = useRef(onClickOutside);

    useEffect(() => {
        resetProjectInEditionRef.current = onClickOutside;
    }, [onClickOutside]);

    useEffect(() => {
        const abortController = new AbortController();

        document.addEventListener(
            'click',
            (event) => {
                if (!textFieldRef.current?.UNSAFE_getDOMNode()?.contains(event.target as Node)) {
                    resetProjectInEditionRef.current();
                }
            },
            { signal: abortController.signal }
        );
        return () => {
            abortController.abort();
        };
    }, [textFieldRef]);
};

export const ProjectEdition = ({ name, onBlur, onResetProjectInEdition }: ProjectEditionProps) => {
    const textFieldRef = useRef<TextFieldRef>(null);
    const [newName, setNewName] = useState<string>(name);

    const handleBlur = () => {
        onBlur(newName);
    };

    const handleKeyDown = (event: KeyboardEvent) => {
        if (event.key === 'Enter') {
            event.preventDefault();

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
            />
        </div>
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
