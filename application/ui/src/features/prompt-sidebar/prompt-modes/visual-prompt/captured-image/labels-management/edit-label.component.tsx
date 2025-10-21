/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CSSProperties, KeyboardEvent, useState } from 'react';

import { ActionButton, ColorPickerDialog, DimensionValue, Flex, TextField } from '@geti/ui';
import { clsx } from 'clsx';

import { Label } from './label.interface';

import classes from './edit-label.module.css';

interface EditLabelProps {
    label: Label;
    onAccept: (editedLabel: Label) => void;
    onClose: () => void;
    isQuiet?: boolean;
    width?: DimensionValue;
}

export const EditLabel = ({ label, onAccept, onClose, isQuiet, width }: EditLabelProps) => {
    const MAX_NAME_LENGTH = 100;
    const [color, setColor] = useState<string>(label.color);
    const [name, setName] = useState<string>(label.name);

    const handleAccept = () => {
        onAccept({ color, name, id: label.id });
        onClose();
    };

    const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleAccept();
        } else if (e.key === 'Escape') {
            onClose();
        }
    };

    return (
        <Flex
            gap={'size-50'}
            width={width}
            justifyContent={'center'}
            alignItems={'center'}
            UNSAFE_className={clsx({ [classes.editLabelContainer]: isQuiet })}
        >
            <ColorPickerDialog
                color={color}
                id={'change-color-button'}
                data-testid={'change-color-button'}
                onColorChange={setColor}
                size={'M'}
                ariaLabelPrefix={'New label'}
            />

            <TextField
                isQuiet={isQuiet}
                flex={1}
                placeholder={'Add label'}
                aria-label={'New label name'}
                data-testid={'new-label-name-input'}
                value={name}
                onChange={setName}
                maxLength={MAX_NAME_LENGTH}
                onKeyDown={(e) => handleKeyDown(e)}
                // eslint-disable-next-line jsx-a11y/no-autofocus
                autoFocus
            />
            <ActionButton
                isQuiet={isQuiet}
                aria-label={'Confirm label'}
                onPress={handleAccept}
                isDisabled={!name.trim()}
                UNSAFE_style={
                    {
                        '--addButtonBgColor': isQuiet ? 'var(--spectrum-global-color-gray-200)' : 'var(--energy-blue)',
                    } as CSSProperties
                }
                UNSAFE_className={classes.plusButton}
            >
                +
            </ActionButton>
        </Flex>
    );
};
