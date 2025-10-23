/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Heading, Text } from '@geti/ui';

import styles from './activate-project-dialog/active-project-dialog.module.scss';

interface CreateProjectConfirmDialogProps {
    isVisible: boolean;
    onClose: () => void;
    onCreate: () => void;
    projectName: string;
    activeProjectName: string;
    isPending: boolean;
}

export const CreateProjectConfirmDialog = ({
    isVisible,
    onCreate,
    projectName,
    activeProjectName,
    onClose,
    isPending,
}: CreateProjectConfirmDialogProps) => {
    return (
        <DialogContainer onDismiss={onClose}>
            {isVisible && (
                <Dialog>
                    <Heading>Create project</Heading>
                    <Divider size={'S'} />
                    <Content>
                        <Text UNSAFE_className={styles.text}>
                            Creating <Text UNSAFE_className={styles.emphasizedText}>{projectName}</Text> will
                            automatically activate it and deactivate the currently active project{' '}
                            <Text UNSAFE_className={styles.emphasizedText}>{activeProjectName}</Text>.
                        </Text>
                    </Content>
                    <ButtonGroup>
                        <Button variant={'secondary'} onPress={onClose}>
                            Cancel
                        </Button>
                        <Button variant={'accent'} onPress={onCreate} isPending={isPending}>
                            Create
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogContainer>
    );
};
