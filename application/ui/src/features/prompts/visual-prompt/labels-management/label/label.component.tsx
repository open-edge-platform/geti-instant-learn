/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ComponentProps, CSSProperties, FormEvent, KeyboardEvent, ReactNode, RefObject } from 'react';

import { ActionButton, ColorPickerDialog, DOMRefValue, Form, TextField, TextFieldRef } from '@geti/ui';

import { MAX_LABEL_NAME_LENGTH } from '../utils';

import styles from './label.module.scss';

interface LabelNameProps {
    name: string;
    onChangeName: (name: string) => void;
    validationError?: string;
    isQuiet?: boolean;
    onClose: () => void;
    ariaLabel: string;
}

const autoFocus = (ref: TextFieldRef<HTMLInputElement> | null) => {
    if (ref === null) return;

    ref.focus();
};

const LabelName = ({ name, onChangeName, isQuiet, onClose, validationError, ariaLabel }: LabelNameProps) => {
    const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Escape') {
            onClose();
        }
    };

    return (
        <TextField
            isQuiet={isQuiet}
            flex={1}
            ref={autoFocus}
            placeholder={'Label name'}
            aria-label={ariaLabel}
            value={name}
            onChange={onChangeName}
            maxLength={MAX_LABEL_NAME_LENGTH}
            onKeyDown={(e) => handleKeyDown(e)}
            errorMessage={validationError}
            validationState={validationError ? 'invalid' : undefined}
            isRequired
        />
    );
};

interface LabelButonProps {
    isDisabled: boolean;
    color: string;
    children: ReactNode;
}

const LabelButon = ({ isDisabled, children, color }: LabelButonProps) => {
    return (
        <ActionButton
            isQuiet
            type={'submit'}
            aria-label={'Confirm label'}
            isDisabled={isDisabled}
            UNSAFE_style={
                {
                    '--labelButtonBgColor': color,
                } as CSSProperties
            }
            UNSAFE_className={styles.labelButton}
        >
            {children}
        </ActionButton>
    );
};

interface LabelColorPickerProps {
    color: string;
    onColorChange: (color: string) => void;
    onOpenChange?: (isOpen: boolean) => void;
}

const LabelColorPicker = ({ color, onColorChange, onOpenChange }: LabelColorPickerProps) => {
    return (
        <ColorPickerDialog
            color={color}
            id={'change-color-button'}
            data-testid={'change-color-button'}
            onColorChange={onColorChange}
            onOpenChange={onOpenChange}
            size={'M'}
        />
    );
};

interface LabelFormProps {
    onSubmit: () => void;
    ref?: RefObject<DOMRefValue<HTMLFormElement> | null> | null;
    children: ComponentProps<typeof Form>['children'];
}

const LabelForm = ({ onSubmit, children, ref }: LabelFormProps) => {
    const handleSubmit = (event: FormEvent) => {
        event.preventDefault();

        onSubmit();
    };

    return (
        <Form validationBehavior={'native'} onSubmit={handleSubmit} ref={ref}>
            {children}
        </Form>
    );
};

export const Label = {
    Form: LabelForm,
    NameField: LabelName,
    ColorPicker: LabelColorPicker,
    Button: LabelButon,
};
