/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CSSProperties, FormEvent, KeyboardEvent, useState } from 'react';

import { LabelType } from '@geti-prompt/api';
import { ActionButton, ColorPickerDialog, DimensionValue, Flex, Form, TextField, TextFieldRef } from '@geti/ui';
import { clsx } from 'clsx';
import { isEmpty } from 'lodash-es';

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

const autoFocus = (ref: TextFieldRef<HTMLInputElement> | null) => {
    if (ref === null) return;

    ref.focus();
};

export const EditLabel = ({ label, onAccept, onClose, isQuiet, width, isDisabled, existingLabels }: EditLabelProps) => {
    const [color, setColor] = useState<string>(label.color);
    const [name, setName] = useState<string>(label.name);

    const validationError = validateLabelName(name, existingLabels, label.id);
    const hasSameName = name.trim() === label.name.trim();
    const hasSameColor = color === label.color;
    const isEditDisabled = !!validationError || isDisabled || isEmpty(name.trim()) || (hasSameName && hasSameColor);

    const handleAccept = (event: FormEvent) => {
        event.preventDefault();

        onAccept({ color, name, id: label.id });
    };

    const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Escape') {
            onClose();
        }
    };

    return (
        <Form validationBehavior={'native'} onSubmit={handleAccept}>
            <Flex
                marginTop={0}
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
                />

                <TextField
                    isQuiet={isQuiet}
                    flex={1}
                    ref={autoFocus}
                    placeholder={'Add label'}
                    aria-label={'New label name'}
                    data-testid={'new-label-name-input'}
                    value={name}
                    onChange={setName}
                    maxLength={MAX_LABEL_NAME_LENGTH}
                    onKeyDown={(e) => handleKeyDown(e)}
                    errorMessage={validationError}
                    validationState={validationError ? 'invalid' : undefined}
                    isRequired
                />
                <ActionButton
                    type={'submit'}
                    isQuiet={isQuiet}
                    aria-label={'Confirm label'}
                    isDisabled={isEditDisabled}
                    UNSAFE_style={
                        {
                            '--addButtonBgColor': isQuiet
                                ? 'var(--spectrum-global-color-gray-200)'
                                : 'var(--energy-blue)',
                        } as CSSProperties
                    }
                    UNSAFE_className={classes.plusButton}
                >
                    +
                </ActionButton>
            </Flex>
        </Form>
    );
};
