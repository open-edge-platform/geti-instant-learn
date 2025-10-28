/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CSSProperties, KeyboardEvent, useState } from 'react';

import { LabelType } from '@geti-prompt/api';
import { ActionButton, ColorPickerDialog, DimensionValue, Flex, TextField } from '@geti/ui';
import { clsx } from 'clsx';

import classes from './edit-label.module.scss';

interface EditLabelProps {
    label: LabelType;
    onAccept: (editedLabel: LabelType) => void;
    onClose: () => void;
    isQuiet?: boolean;
    width?: DimensionValue;
    isDisabled?: boolean;
    existingLabelsNames: string[];
}

const isUniqueName = (name: string, existingLabelsNames: string[]) => {
    return !existingLabelsNames.includes(name);
};

export const EditLabel = ({
    label,
    onAccept,
    onClose,
    isQuiet,
    width,
    isDisabled,
    existingLabelsNames,
}: EditLabelProps) => {
    const MAX_NAME_LENGTH = 100;
    const [color, setColor] = useState<string>(label.color);
    const [name, setName] = useState<string>(label.name);

    const isEditDisabled = !isUniqueName(name, existingLabelsNames) || !name.trim() || isDisabled;

    const handleAccept = () => {
        onAccept({ color, name, id: label.id });
        onClose();
    };

    const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Enter' && !isEditDisabled) {
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
                validate={(newName) => isUniqueName(newName, existingLabelsNames) || 'Label name must be unique.'}
            />
            <ActionButton
                isQuiet={isQuiet}
                aria-label={'Confirm label'}
                onPress={handleAccept}
                isDisabled={isEditDisabled}
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
