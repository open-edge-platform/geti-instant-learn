/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CSSProperties, KeyboardEvent, useState } from 'react';

import { LabelType } from '@geti-prompt/api';
import { ActionButton, ColorPickerDialog, DimensionValue, Flex, TextField } from '@geti/ui';
import { clsx } from 'clsx';

import { MAX_LABEL_NAME_LENGTH, validateLabelName } from '../utils';

import classes from './edit-label.module.scss';

interface EditLabelProps {
    label: LabelType;
    onAccept: (editedLabel: LabelType) => void;
    onClose: () => void;
    isQuiet?: boolean;
    width?: DimensionValue;
    isDisabled?: boolean;
    existingLabels: LabelType[];
}

export const EditLabel = ({ label, onAccept, onClose, isQuiet, width, isDisabled, existingLabels }: EditLabelProps) => {
    const [color, setColor] = useState<string>(label.color);
    const [name, setName] = useState<string>(label.name);

    const validationError = validateLabelName(name, existingLabels, label.id);
    const hasSameName = name.trim() === label.name.trim();
    const isEditDisabled = !!validationError || isDisabled || hasSameName;

    const handleAccept = () => {
        if (isEditDisabled) return;

        onAccept({ color, name, id: label.id });
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
                maxLength={MAX_LABEL_NAME_LENGTH}
                onKeyDown={(e) => handleKeyDown(e)}
                // eslint-disable-next-line jsx-a11y/no-autofocus
                autoFocus
                validationState={validationError ? 'invalid' : 'valid'}
                errorMessage={validationError}
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
