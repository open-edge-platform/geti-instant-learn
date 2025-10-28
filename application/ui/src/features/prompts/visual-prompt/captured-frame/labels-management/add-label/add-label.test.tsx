/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, type RenderOptions } from '@geti-prompt/test-utils';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { paths } from '../../../../../../routes/paths';
import { http, server } from '../../../../../../setup-test';
import { AddLabel } from './add-label.component';

class AddLabelPageObject {
    showDialog() {
        fireEvent.click(screen.getByRole('button', { name: /add label/i }));
    }

    getDialog() {
        return screen.queryByRole('dialog');
    }

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

        await userEvent.type(input, text);
    }

    async clickConfirmButton() {
        fireEvent.click(this.getConfirmLabelButton());
    }

    async closeDialogWithKeyboard() {
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

const renderAddLabel = ({
    existingLabelsNames = [],
    options,
}: { existingLabelsNames?: string[]; options?: RenderOptions } = {}) => {
    const result = render(<AddLabel existingLabelsNames={existingLabelsNames} />, options);

    return {
        result,
        addLabelPage: new AddLabelPageObject(),
    };
};

describe('AddLabel', () => {
    it('creates a new label when valid name is entered and confirmed', async () => {
        const projectId = 'test-project-id';
        let newLabelBody: { id?: string; name: string; color?: string | null } | null = null;

        server.use(
            http.post('/api/v1/projects/{project_id}/labels', async ({ request, params }) => {
                expect(params.project_id).toBe(projectId);

                newLabelBody = await request.clone().json();

                return HttpResponse.json(newLabelBody, { status: 201 });
            })
        );

        const { addLabelPage } = renderAddLabel({
            options: {
                route: paths.project({ projectId }),
                path: paths.project.pattern,
            },
        });

        addLabelPage.showDialog();
        await addLabelPage.typeInNameInput('New Label');

        expect(addLabelPage.getConfirmLabelButton()).toBeEnabled();

        await addLabelPage.clickConfirmButton();

        await waitFor(() => {
            expect(newLabelBody).not.toBeNull();

            expect(newLabelBody?.name).toBe('New Label');
            expect(newLabelBody?.id).toBeTruthy();
            expect(newLabelBody?.color).toBeTruthy();
        });

        await waitFor(() => {
            expect(addLabelPage.getDialog()).not.toBeInTheDocument();
        });
    });

    it('creates a new label when valid name is entered and confirmed using enter key', async () => {
        const projectId = 'test-project-id';
        let newLabelBody: { id?: string; name: string; color?: string | null } | null = null;

        server.use(
            http.post('/api/v1/projects/{project_id}/labels', async ({ request, params }) => {
                expect(params.project_id).toBe(projectId);

                newLabelBody = await request.clone().json();

                return HttpResponse.json(newLabelBody, { status: 201 });
            })
        );

        const { addLabelPage } = renderAddLabel({
            options: {
                route: paths.project({ projectId }),
                path: paths.project.pattern,
            },
        });

        addLabelPage.showDialog();
        await addLabelPage.typeInNameInput('New Label');

        expect(addLabelPage.getConfirmLabelButton()).toBeEnabled();

        await addLabelPage.confirmLabelWithKeyboard();

        await waitFor(() => {
            expect(newLabelBody).not.toBeNull();
            expect(newLabelBody?.name).toBe('New Label');
            expect(newLabelBody?.id).toBeTruthy();
            expect(newLabelBody?.color).toBeTruthy();
        });

        await waitFor(() => {
            expect(addLabelPage.getDialog()).not.toBeInTheDocument();
        });
    });

    it('does not creates a new label when Enter is pressed with duplicate name', async () => {
        const existingLabelsNames = ['Duplicate'];
        let requestReceived = false;

        server.use(
            http.post('/api/v1/projects/{project_id}/labels', async () => {
                requestReceived = true;
                return HttpResponse.json({}, { status: 201 });
            })
        );

        const { addLabelPage } = renderAddLabel({ existingLabelsNames });

        addLabelPage.showDialog();
        await addLabelPage.typeInNameInput(existingLabelsNames[0]);
        await addLabelPage.confirmLabelWithKeyboard();

        // Wait a bit to ensure no request was made
        await new Promise((resolve) => setTimeout(resolve, 100));

        expect(requestReceived).toBe(false);
    });

    it('closes dialog when Escape key is pressed', async () => {
        const { addLabelPage } = renderAddLabel();

        addLabelPage.showDialog();

        await addLabelPage.closeDialogWithKeyboard();

        await waitFor(() => {
            expect(addLabelPage.getNameInput()).not.toBeInTheDocument();
        });
    });

    it('autofocuses the name input when dialog opens', async () => {
        const { addLabelPage } = renderAddLabel();

        addLabelPage.showDialog();

        const input = addLabelPage.getNameInput();
        await waitFor(() => {
            expect(input).toHaveFocus();
        });
    });

    it('disables confirm button when name is empty', async () => {
        const { addLabelPage } = renderAddLabel();

        addLabelPage.showDialog();

        expect(addLabelPage.getNameInput()).toHaveValue('');
        expect(addLabelPage.getConfirmLabelButton()).toBeDisabled();
    });

    it('disables confirm button when name contains only whitespace', async () => {
        const emptyName = '   ';

        const { addLabelPage } = renderAddLabel();

        addLabelPage.showDialog();
        await addLabelPage.typeInNameInput(emptyName);

        expect(addLabelPage.getNameInput()).toHaveValue(emptyName);
        expect(addLabelPage.getConfirmLabelButton()).toBeDisabled();
    });

    it('disables confirm button for duplicate label name', async () => {
        const existingLabelsNames = ['Existing Label', 'Another Label'];
        const { addLabelPage } = renderAddLabel({ existingLabelsNames });

        addLabelPage.showDialog();
        await addLabelPage.typeInNameInput(existingLabelsNames[0]);

        expect(addLabelPage.getNameInput()).toHaveValue(existingLabelsNames[0]);
        expect(addLabelPage.getConfirmLabelButton()).toBeDisabled();
    });

    it('shows validation error for duplicate label name', async () => {
        const existingLabelsNames = ['Existing Label', 'Another Label'];
        const { addLabelPage } = renderAddLabel({ existingLabelsNames });

        addLabelPage.showDialog();
        await addLabelPage.typeInNameInput(existingLabelsNames[0]);

        expect(addLabelPage.getNameInput()).toHaveValue(existingLabelsNames[0]);
        expect(addLabelPage.getValidationError()).toBeInTheDocument();
    });
});
