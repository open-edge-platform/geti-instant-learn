/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Heading, Text } from '@geti/ui';

import styles from './active-project-dialog.module.scss';

interface ActivateProjectDialogProps {
    isVisible: boolean;
    onClose: () => void;
    activeProjectName: string;
    inactiveProjectName: string;
    onActivate: () => void;
}

export const ActivateProjectDialog = ({
    isVisible,
    onActivate,
    inactiveProjectName,
    activeProjectName,
    onClose,
}: ActivateProjectDialogProps) => {
    return (
        <DialogContainer onDismiss={onClose}>
            {isVisible && (
                <Dialog>
                    <Heading>Activate project</Heading>
                    <Divider size={'S'} />
                    <Content>
                        <Text UNSAFE_className={styles.text}>
                            You are about to activate the{' '}
                            <Text UNSAFE_className={styles.emphasizedText}>{inactiveProjectName}</Text> which is
                            currently inactive. <br />
                            Activating the{' '}
                            <Text UNSAFE_className={styles.emphasizedText}>
                                {inactiveProjectName} will deactivate
                            </Text>{' '}
                            the project <Text UNSAFE_className={styles.emphasizedText}>{activeProjectName}</Text>.
                        </Text>
                    </Content>
                    <ButtonGroup>
                        <Button variant={'secondary'} onPress={onClose}>
                            Cancel
                        </Button>
                        <Button variant={'accent'} onPress={onActivate}>
                            Activate
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogContainer>
    );
};
