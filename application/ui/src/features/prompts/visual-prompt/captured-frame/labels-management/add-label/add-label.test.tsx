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

    async getDialog() {
        return screen.findByRole('dialog');
    }

    getNameInput() {
        return screen.queryByLabelText(/new label name/i);
    }

    focusNameInput() {
        const input = this.getNameInput();
        input != null && fireEvent.click(input);
    }

    getConfirmButton() {
        return screen.queryByLabelText(/confirm label/i);
    }

    getConfirmLabelButton() {
        return screen.getByRole('button', { name: /confirm label/i });
    }

    getColorButton() {
        return screen.queryByTestId('change-color-button');
    }

    async typeInNameInput(text: string) {
        const input = this.getNameInput();
        input != null && (await userEvent.type(input, text));
    }

    async clickConfirmButton() {
        fireEvent.click(this.getConfirmLabelButton());
    }

    async closeDialog() {
        await userEvent.keyboard('{escape}');
    }

    async pressKey(key: string) {
        await userEvent.keyboard(`{${key}}`);
    }

    async getValidationError() {
        return screen.findByText(/label's name must be unique/i);
    }

    isConfirmButtonDisabled() {
        const confirmButton = this.getConfirmButton();
        return confirmButton?.hasAttribute('disabled');
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
        });

        await waitFor(() => {
            expect(newLabelBody?.name).toBe('New Label');
            expect(newLabelBody?.id).toBeTruthy();
            expect(newLabelBody?.color).toBeTruthy();
        });
    });

    it('closes dialog when Escape key is pressed', async () => {
        const { addLabelPage } = renderAddLabel();

        addLabelPage.showDialog();
        addLabelPage.focusNameInput();

        await addLabelPage.closeDialog();

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

    /*describe('Label Creation - Happy Path', () => {
        it('creates label when Enter key is pressed', async () => {
            const projectId = 'test-project-id';
            let requestReceived = false;

            server.use(
                http.post(
                    '/api/v1/projects/{project_id}/labels',
                    async ({ request }: { request: StrictRequest<DefaultBodyType> }) => {
                        const body = await request.json();
                        requestReceived = true;
                        return HttpResponse.json(body, { status: 201 });
                    }
                )
            );

            const { addLabelPage } = renderAddLabel({ projectId });

            await addLabelPage.showDialog();
            await addLabelPage.typeInNameInput('Quick Label');
            await addLabelPage.pressKey('Enter');

            await waitFor(() => {
                expect(requestReceived).toBe(true);
            });
        });

        it('closes dialog after successful label creation', async () => {
            server.use(
                http.post(
                    '/api/v1/projects/{project_id}/labels',
                    async ({ request }: { request: StrictRequest<DefaultBodyType> }) => {
                        const body = await request.json();
                        return HttpResponse.json(body, { status: 201 });
                    }
                )
            );

            const { addLabelPage } = renderAddLabel();

            await addLabelPage.showDialog();
            await addLabelPage.typeInNameInput('New Label');
            await addLabelPage.clickConfirmButton();

            await waitFor(() => {
                expect(addLabelPage.getNameInput()).not.toBeInTheDocument();
            });
        });
    });

    describe('Validation - Edge Cases', () => {
        it('disables confirm button when name is empty', async () => {
            const { addLabelPage } = renderAddLabel();

            await addLabelPage.showDialog();
            await addLabelPage.findNameInput();

            expect(addLabelPage.isConfirmButtonDisabled()).toBe(true);
        });

        it('disables confirm button when name contains only whitespace', async () => {
            const { addLabelPage } = renderAddLabel();

            await addLabelPage.showDialog();
            await addLabelPage.typeInNameInput('   ');

            expect(addLabelPage.isConfirmButtonDisabled()).toBe(true);
        });

        it('shows validation error for duplicate label name', async () => {
            const existingLabelsNames = ['Existing Label', 'Another Label'];
            const { addLabelPage } = renderAddLabel({ existingLabelsNames });

            await addLabelPage.showDialog();
            await addLabelPage.typeInNameInput('Existing Label');

            const error = await addLabelPage.getValidationError();
            expect(error).toBeInTheDocument();
        });

        it('disables confirm button for duplicate label name', async () => {
            const existingLabelsNames = ['Existing Label'];
            const { addLabelPage } = renderAddLabel({ existingLabelsNames });

            await addLabelPage.showDialog();
            await addLabelPage.typeInNameInput('Existing Label');

            expect(addLabelPage.isConfirmButtonDisabled()).toBe(true);
        });

        it('does not submit when Enter is pressed with duplicate name', async () => {
            const existingLabelsNames = ['Duplicate'];
            let requestReceived = false;

            server.use(
                http.post('/api/v1/projects/{project_id}/labels', async () => {
                    requestReceived = true;
                    return HttpResponse.json({}, { status: 201 });
                })
            );

            const { addLabelPage } = renderAddLabel({ existingLabelsNames });

            await addLabelPage.showDialog();
            await addLabelPage.typeInNameInput('Duplicate');
            await addLabelPage.pressKey('Enter');

            // Wait a bit to ensure no request was made
            await new Promise((resolve) => setTimeout(resolve, 100));

            expect(requestReceived).toBe(false);
        });

        it('allows label creation with unique name', async () => {
            const existingLabelsNames = ['Existing Label'];
            let capturedRequest: { id: string; name: string; color: string } | null = null;

            server.use(
                http.post(
                    '/api/v1/projects/{project_id}/labels',
                    async ({ request }: { request: StrictRequest<DefaultBodyType> }) => {
                        capturedRequest = (await request.json()) as { id: string; name: string; color: string };
                        return HttpResponse.json(capturedRequest, { status: 201 });
                    }
                )
            );

            const { addLabelPage } = renderAddLabel({ existingLabelsNames });

            await addLabelPage.showDialog();
            await addLabelPage.typeInNameInput('Unique Label');
            await addLabelPage.clickConfirmButton();

            await waitFor(() => {
                expect(capturedRequest).toBeTruthy();
                expect(capturedRequest?.name).toBe('Unique Label');
            });
        });

        it('respects maximum name length', async () => {
            const { addLabelPage } = renderAddLabel();

            await addLabelPage.showDialog();

            const longName = 'a'.repeat(150); // Exceeds the 100 character limit
            await addLabelPage.typeInNameInput(longName);

            const input = await addLabelPage.findNameInput();
            expect((input as HTMLInputElement).value.length).toBeLessThanOrEqual(100);
        });
    });

    describe('Color Selection', () => {
        it('includes a color in the created label', async () => {
            let capturedRequest: { id: string; name: string; color: string } | null = null;

            server.use(
                http.post(
                    '/api/v1/projects/{project_id}/labels',
                    async ({ request }: { request: StrictRequest<DefaultBodyType> }) => {
                        capturedRequest = (await request.json()) as { id: string; name: string; color: string };
                        return HttpResponse.json(capturedRequest, { status: 201 });
                    }
                )
            );

            const { addLabelPage } = renderAddLabel();

            await addLabelPage.showDialog();
            await addLabelPage.typeInNameInput('Colored Label');
            await addLabelPage.clickConfirmButton();

            await waitFor(() => {
                expect(capturedRequest).toBeTruthy();
                expect(capturedRequest?.color).toBeTruthy();
                expect(typeof capturedRequest?.color).toBe('string');
            });
        });

        it('renders color picker button in dialog', async () => {
            const { addLabelPage } = renderAddLabel();

            await addLabelPage.showDialog();

            expect(addLabelPage.getColorButton()).toBeInTheDocument();
        });
    });

    describe('Pending State', () => {
        it('disables confirm button while mutation is pending', async () => {
            let resolveRequest: ((value: unknown) => void) | undefined;
            const requestPromise = new Promise((resolve) => {
                resolveRequest = resolve;
            });

            server.use(
                http.post(
                    '/api/v1/projects/{project_id}/labels',
                    async ({ request }: { request: StrictRequest<DefaultBodyType> }) => {
                        const body = await request.json();
                        await requestPromise;
                        return HttpResponse.json(body, { status: 201 });
                    }
                )
            );

            const { addLabelPage } = renderAddLabel();

            await addLabelPage.showDialog();
            await addLabelPage.typeInNameInput('Test Label');
            await addLabelPage.clickConfirmButton();

            // Button should be disabled while request is pending
            await waitFor(() => {
                expect(addLabelPage.isConfirmButtonDisabled()).toBe(true);
            });

            // Resolve the request
            if (resolveRequest) {
                resolveRequest(null);
            }

            // Dialog should eventually close
            await waitFor(() => {
                expect(addLabelPage.getNameInput()).not.toBeInTheDocument();
            });
        });
    });

    describe('API Error Handling', () => {
        it('handles API errors gracefully', async () => {
            server.use(
                http.post('/api/v1/projects/{project_id}/labels', () => {
                    return HttpResponse.json({ error: 'Internal Server Error' }, { status: 500 });
                })
            );

            const { addLabelPage } = renderAddLabel();

            await addLabelPage.showDialog();
            await addLabelPage.typeInNameInput('Test Label');
            await addLabelPage.clickConfirmButton();

            // Dialog should remain open on error
            await waitFor(() => {
                expect(addLabelPage.getNameInput()).toBeInTheDocument();
            });
        });
    });

    describe('Multiple Labels', () => {
        it('handles multiple existing labels correctly', async () => {
            const existingLabelsNames = ['Label 1', 'Label 2', 'Label 3', 'Label 4'];
            const { addLabelPage } = renderAddLabel({ existingLabelsNames });

            await addLabelPage.showDialog();

            // Should not allow any of the existing names
            for (const existingName of existingLabelsNames) {
                await addLabelPage.typeInNameInput(existingName);
                expect(addLabelPage.isConfirmButtonDisabled()).toBe(true);
                await userEvent.clear(await addLabelPage.findNameInput());
            }

            // Should allow a new unique name
            await addLabelPage.typeInNameInput('Label 5');
            expect(addLabelPage.isConfirmButtonDisabled()).toBe(false);
        });
    });*/
});
