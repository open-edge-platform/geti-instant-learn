/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Flex } from '@geti/ui';
import { Redo, Undo } from '@geti/ui/icons';

import { useUndoRedo } from '../../../annotator/undo-redo/undo-redo-provider.component';

export const UndoRedo = ({ isDisabled }: { isDisabled?: boolean }) => {
    const { undo, canUndo, redo, canRedo } = useUndoRedo();

    return (
        <Flex alignItems='center' justify-content='center' data-testid='undo-redo-tools'>
            <ActionButton
                isQuiet
                id='undo-button'
                data-testid='undo-button'
                onPress={undo}
                aria-label='undo'
                isDisabled={!canUndo || isDisabled}
            >
                <Undo />
            </ActionButton>

            <ActionButton
                isQuiet
                id='redo-button'
                data-testid='redo-button'
                aria-label='redo'
                onPress={redo}
                isDisabled={!canRedo || isDisabled}
            >
                <Redo />
            </ActionButton>
        </Flex>
    );
};
