/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { LabelType } from '@geti-prompt/api';
import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';

import { EditLabel } from './edit-label.component';

class EditLabelPageObject {
    getNameInput() {
        return screen.queryByLabelText(/new label name/i);
    }

    getConfirmLabelButton() {
        return screen.getByRole('button', { name: /confirm label/i });
    }

    async typeInNameInput(text: string) {
        const input = this.getNameInput();

        if (input === null) {
            return;
        }

        await userEvent.clear(input);
        await userEvent.type(input, text);
    }

    async clickConfirmButton() {
        fireEvent.click(this.getConfirmLabelButton());
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
    existingLabelsNames = [],
    onAccept = () => {},
    onClose = () => {},
    isDisabled = false,
}: {
    label?: LabelType;
    onAccept?: (label: LabelType) => void;
    onClose?: () => void;
    existingLabelsNames?: string[];
    isDisabled?: boolean;
} = {}) => {
    const result = render(
        <EditLabel
            label={label}
            onAccept={onAccept}
            onClose={onClose}
            existingLabelsNames={existingLabelsNames}
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
        const existingLabelsNames = ['Another Label'];
        const { editLabelPage } = renderEditLabel({ existingLabelsNames });

        await editLabelPage.typeInNameInput(existingLabelsNames[0]);

        expect(editLabelPage.getNameInput()).toHaveValue(existingLabelsNames[0]);
        expect(editLabelPage.getConfirmLabelButton()).toBeDisabled();
    });

    it('disables confirm button when new name is the same as the old one', async () => {
        const labelName = 'Existing Label';
        const { editLabelPage } = renderEditLabel({ label: { id: '1', name: labelName, color: '#000000' } });

        await editLabelPage.typeInNameInput(labelName);

        expect(editLabelPage.getNameInput()).toHaveValue(labelName);
        expect(editLabelPage.getConfirmLabelButton()).toBeDisabled();
    });

    it('disables confirm button when isDisabled is true', async () => {
        const { editLabelPage } = renderEditLabel({ isDisabled: true });

        expect(editLabelPage.getConfirmLabelButton()).toBeDisabled();
    });

    it('shows an error message when the name is duplicate', async () => {
        const existingLabelsNames = ['Another Label'];
        const { editLabelPage } = renderEditLabel({ existingLabelsNames });

        await editLabelPage.typeInNameInput(existingLabelsNames[0]);

        expect(editLabelPage.getNameInput()).toHaveValue(existingLabelsNames[0]);
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
