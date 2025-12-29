/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { LabelType } from '@geti-prompt/api';
import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';

import { EditLabel } from './edit-label.component';

class EditLabelPageObject {
    getNameInput() {
        return screen.queryByLabelText(/label name/i);
    }

    getConfirmLabelButton() {
        return screen.getByRole('button', { name: /confirm label/i });
    }

    getColorPickerButton() {
        return screen.getByRole('button', { name: 'Color picker button' });
    }

    async typeInNameInput(text: string) {
        const input = this.getNameInput();

        if (input === null) {
            return;
        }

        await userEvent.clear(input);
        await userEvent.type(input, text);
    }

    clickConfirmButton() {
        fireEvent.click(this.getConfirmLabelButton());
    }

    clickColorPickerButton() {
        fireEvent.click(this.getColorPickerButton());
    }

    async changeColor(color: string) {
        const input = screen.getByTestId('change-color-button-color-input');
        await userEvent.clear(input);
        await userEvent.type(input, color);

        fireEvent.click(within(screen.getByTestId('modal')).getByRole('button', { name: 'Confirm' }));
    }

    async closeWithKeyboard() {
        const input = this.getNameInput();

        if (input === null) {
            return;
        }

        await userEvent.type(input, '{Escape}');
    }

    async confirmLabelWithKeyboard() {
        const input = this.getNameInput();

        if (input === null) {
            return;
        }

        await userEvent.type(input, '{Enter}');
    }

    getValidationError() {
        return screen.getByText(/label name must be unique/i);
    }
}

const renderEditLabel = ({
    label = { id: '1', name: 'Existing Label', color: '#000000' },
    existingLabels = [],
    onAccept = () => {},
    onClose = () => {},
    isDisabled = false,
}: {
    label?: LabelType;
    onAccept?: (label: LabelType) => void;
    onClose?: () => void;
    existingLabels?: LabelType[];
    isDisabled?: boolean;
} = {}) => {
    const result = render(
        <EditLabel
            label={label}
            onAccept={onAccept}
            onClose={onClose}
            existingLabels={existingLabels}
            isDisabled={isDisabled}
        />
    );

    return {
        result,
        editLabelPage: new EditLabelPageObject(),
    };
};

describe('EditLabel', () => {
    it('disables confirm button when the name contains whitespaces', async () => {
        const { editLabelPage } = renderEditLabel();

        await editLabelPage.typeInNameInput(' ');

        expect(editLabelPage.getNameInput()).toHaveValue(' ');
        expect(editLabelPage.getConfirmLabelButton()).toBeDisabled();
    });

    it('disables confirm button when the name is duplicate', async () => {
        const existingLabels: LabelType[] = [{ id: '2', name: 'Another Label', color: '#000000' }];
        const { editLabelPage } = renderEditLabel({ existingLabels });

        await editLabelPage.typeInNameInput(existingLabels[0].name);

        expect(editLabelPage.getNameInput()).toHaveValue(existingLabels[0].name);
        expect(editLabelPage.getConfirmLabelButton()).toBeDisabled();
    });

    it('enables confirm button when the color is changed', async () => {
        const existingLabels: LabelType[] = [{ id: '2', name: 'Another Label', color: '#000000' }];
        const mockOnAccept = vi.fn();
        const { editLabelPage } = renderEditLabel({ existingLabels, onAccept: mockOnAccept });

        editLabelPage.clickColorPickerButton();
        await editLabelPage.changeColor('#FFFFFF');

        expect(editLabelPage.getConfirmLabelButton()).toBeEnabled();
        editLabelPage.clickConfirmButton();

        expect(mockOnAccept).toHaveBeenCalledWith(expect.objectContaining({ color: '#FFFFFF' }));
    });

    it('does not invoke onAccept when new name is the same as the old one', async () => {
        const onAccept = vi.fn();
        const labelName = 'Existing Label';
        const existingLabels: LabelType[] = [{ id: '2', name: labelName, color: '#000000' }];

        const { editLabelPage } = renderEditLabel({
            existingLabels,
            onAccept,
        });

        await editLabelPage.typeInNameInput(labelName);

        expect(editLabelPage.getNameInput()).toHaveValue(labelName);
        expect(onAccept).not.toHaveBeenCalled();
    });

    it('disables confirm button when isDisabled is true', async () => {
        const { editLabelPage } = renderEditLabel({ isDisabled: true });

        expect(editLabelPage.getConfirmLabelButton()).toBeDisabled();
    });

    it('shows an error message when the name is duplicate', async () => {
        const existingLabels: LabelType[] = [{ id: '2', name: 'Another Label', color: '#000000' }];
        const { editLabelPage } = renderEditLabel({ existingLabels });

        await editLabelPage.typeInNameInput(existingLabels[0].name);

        expect(editLabelPage.getNameInput()).toHaveValue(existingLabels[0].name);
        expect(editLabelPage.getValidationError()).toBeInTheDocument();
    });

    it('invokes onAccept when the confirm button is clicked', async () => {
        const onAccept = vi.fn();
        const label = { id: '1', name: 'Existing Label', color: '#000000' };
        const newLabelName = 'New Label';

        const { editLabelPage } = renderEditLabel({ onAccept, label });

        await editLabelPage.typeInNameInput(newLabelName);
        await editLabelPage.clickConfirmButton();

        expect(onAccept).toHaveBeenCalledWith({
            ...label,
            name: newLabelName,
        });
    });

    it('invokes onAccept when an enter key is pressed', async () => {
        const onAccept = vi.fn();
        const label = { id: '1', name: 'Existing Label', color: '#000000' };
        const newLabelName = 'New Label';

        const { editLabelPage } = renderEditLabel({ onAccept, label });

        await editLabelPage.typeInNameInput(newLabelName);
        await editLabelPage.confirmLabelWithKeyboard();

        expect(onAccept).toHaveBeenCalledWith({
            ...label,
            name: newLabelName,
        });
    });

    it('does not invoke onAccept with enter key when the confirm button is disabled', async () => {
        const onAccept = vi.fn();
        const label = { id: '1', name: 'Existing Label', color: '#000000' };
        const newLabelName = 'New Label';

        const { editLabelPage } = renderEditLabel({ onAccept, label, isDisabled: true });

        await editLabelPage.typeInNameInput(newLabelName);
        await editLabelPage.confirmLabelWithKeyboard();

        expect(onAccept).not.toHaveBeenCalled();
    });

    it('invokes onClose when the cancel button is clicked', async () => {
        const onClose = vi.fn();
        const { editLabelPage } = renderEditLabel({ onClose });

        await editLabelPage.closeWithKeyboard();
        expect(onClose).toHaveBeenCalled();
    });
});
