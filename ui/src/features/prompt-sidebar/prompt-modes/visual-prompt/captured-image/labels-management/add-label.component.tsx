/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { ActionButton, Button, ColorPickerDialog, Content, Dialog, DialogTrigger, Flex, TextField } from '@geti/ui';
import { v4 as uuid } from 'uuid';

import { Label } from './label.interface';

interface AddLabelProps {
    addLabel: (label: Label) => void;
}

export const AddLabel = ({ addLabel }: AddLabelProps) => {
    //TODO: prevent adding label with same name (?)
    //TODO: add color picker default color - find out what color it should be

    const DEFAULT_COLOR = '#ededed';
    const MAX_NAME_LENGTH = 100;

    const [color, setColor] = useState<string>(DEFAULT_COLOR);
    const [name, setName] = useState<string>('');

    const handleAddingLabel = (closeDialog: () => void) => {
        addLabel({ name, color, id: uuid() });
        closeDialog();
    };

    const handleKeyDown = (e: React.KeyboardEvent, close: () => void) => {
        if (e.key === 'Enter') {
            handleAddingLabel(close);
        } else e.key === 'Escape' && close();
    };

    const onDialogClose = (isOpen: boolean) => {
        if (!isOpen) {
            setName('');
            setColor(DEFAULT_COLOR);
        }
    };

    return (
        <DialogTrigger type={'popover'} hideArrow placement={'bottom right'} onOpenChange={onDialogClose}>
            <Button variant={'secondary'} UNSAFE_style={{ border: 'none' }}>
                Add label
            </Button>
            {(_close) => (
                <Dialog>
                    <Content>
                        <Flex gap={'size-50'}>
                            <ColorPickerDialog
                                color={color}
                                id={'change-color-button'}
                                data-testid={'change-color-button'}
                                onColorChange={setColor}
                                size={'M'}
                                ariaLabelPrefix={'new label'}
                            />
                            <TextField
                                flex={1}
                                placeholder={'Add label'}
                                aria-label={'New label name'}
                                data-testid={'new-label-name-input'}
                                value={name}
                                onChange={setName}
                                maxLength={MAX_NAME_LENGTH}
                                onKeyDown={(e) => handleKeyDown(e, _close)}
                            ></TextField>
                            <ActionButton onPress={() => handleAddingLabel(_close)} isDisabled={!name.trim()}>
                                +
                            </ActionButton>
                        </Flex>
                    </Content>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
