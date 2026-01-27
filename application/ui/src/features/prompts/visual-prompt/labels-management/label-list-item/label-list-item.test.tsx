/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { LabelType } from '@/api';
import { getMockedLabel, render } from '@/test-utils';
import { fireEvent, screen, waitFor, waitForElementToBeRemoved } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { beforeEach, vi } from 'vitest';

import { http, server } from '../../../../../setup-test';
import { SelectedFrameProvider } from '../../../../../shared/selected-frame-provider.component';
import { useVisualPrompt, VisualPromptProvider } from '../../visual-prompt-provider.component';
import { LabelListItem } from './label-list-item.component';

const mockDeleteAnnotationByLabelId = vi.fn();

vi.mock('../../../../annotator/providers/annotation-actions-provider.component', () => ({
    useAnnotationActions: () => ({
        deleteAnnotationByLabelId: mockDeleteAnnotationByLabelId,
    }),
}));

const App = ({
    label,
    existingLabels,
    isSelected,
    onSelect,
}: {
    label: LabelType;
    existingLabels: LabelType[];
    isSelected: boolean;
    onSelect: () => void;
}) => {
    const { selectedLabelId } = useVisualPrompt();

    return (
        <>
            <span aria-label={'Selected label id'}>{selectedLabelId ?? 'Empty'}</span>
            <LabelListItem label={label} onSelect={onSelect} isSelected={isSelected} existingLabels={existingLabels} />
        </>
    );
};

const renderLabelListItem = async ({
    label = getMockedLabel(),
    existingLabels = [],
    isSelected = true,
    onSelect = vi.fn(),
}: {
    label?: LabelType;
    existingLabels?: LabelType[];
    isSelected?: boolean;
    onSelect?: () => void;
}) => {
    render(
        <SelectedFrameProvider>
            <VisualPromptProvider>
                <App label={label} onSelect={onSelect} isSelected={isSelected} existingLabels={existingLabels} />
            </VisualPromptProvider>
        </SelectedFrameProvider>
    );

    await waitForElementToBeRemoved(screen.getByRole('progressbar'));
};

describe('LabelListItem', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('deletes label and resets selected label', async () => {
        let labelIdToBeRemoved: string | null = null;
        const label = {
            id: '123',
            name: 'label',
            color: '#000000',
        };

        server.use(
            http.delete('/api/v1/projects/{project_id}/labels/{label_id}', ({ params, response }) => {
                labelIdToBeRemoved = params.label_id;
                // @ts-expect-error Issue in openapi-types
                return response(200).json({});
            }),
            http.get('/api/v1/projects/{project_id}/prompts/{prompt_id}', () => {
                return HttpResponse.json({}, { status: 200 });
            }),
            http.get('/api/v1/projects/{project_id}/labels', () => {
                return HttpResponse.json({
                    labels: [label],
                    pagination: {
                        total: 1,
                        count: 1,
                        offset: 0,
                        limit: 10,
                    },
                });
            })
        );

        await renderLabelListItem({ label, onSelect: vi.fn(), isSelected: true, existingLabels: [] });

        expect(screen.getByLabelText('Selected label id')).toHaveTextContent(label.id);

        await userEvent.hover(screen.getByLabelText(`Label ${label.name}`));

        fireEvent.click(await screen.findByRole('button', { name: `Delete ${label.name} label` }));

        await waitFor(() => {
            expect(labelIdToBeRemoved).toBe(label.id);
            expect(screen.getByLabelText('Selected label id')).toHaveTextContent('Empty');
            expect(mockDeleteAnnotationByLabelId).toHaveBeenCalledWith(label.id);
        });
    });

    it('selected label has proper selected styles', async () => {
        const label = {
            id: '123',
            name: 'label',
            color: '#000000',
        };

        await renderLabelListItem({ label, onSelect: vi.fn(), isSelected: true, existingLabels: [] });

        expect(screen.getByLabelText(`Label ${label.name}`)).toHaveAttribute('aria-selected', 'true');
        expect(screen.getByLabelText(`Label ${label.name}`)).toHaveClass(/selected/);
    });

    it('not selected label has no selected styles', async () => {
        const label = {
            id: '123',
            name: 'label',
            color: '#000000',
        };

        await renderLabelListItem({ label, onSelect: vi.fn(), isSelected: false, existingLabels: [] });

        expect(screen.getByLabelText(`Label ${label.name}`)).toHaveAttribute('aria-selected', 'false');
        expect(screen.getByLabelText(`Label ${label.name}`)).not.toHaveClass(/selected/);
    });
});
