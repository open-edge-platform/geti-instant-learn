/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';

import { http, server } from '../../../../../setup-test';
import { LabelListItem } from './label-list-item.component';

describe('LabelListItem', () => {
    it('deletes label', async () => {
        let labelIdToBeRemoved: string | null = null;
        server.use(
            http.delete('/api/v1/projects/{project_id}/labels/{label_id}', ({ params, response }) => {
                labelIdToBeRemoved = params.label_id;
                // @ts-expect-error Issue in openapi-types
                return response(200).json({});
            })
        );

        const label = {
            id: '123',
            name: 'label',
            color: '#000000',
        };

        render(<LabelListItem label={label} onSelect={vi.fn()} isSelected existingLabels={[]} />);

        await userEvent.hover(screen.getByLabelText(`Label ${label.name}`));

        fireEvent.click(await screen.findByRole('button', { name: `Delete ${label.name} label` }));

        await waitFor(() => {
            expect(labelIdToBeRemoved).toBe(label.id);
        });
    });

    it('selected label has proper selected styles', async () => {
        const label = {
            id: '123',
            name: 'label',
            color: '#000000',
        };

        render(<LabelListItem label={label} onSelect={vi.fn()} isSelected existingLabels={[]} />);

        expect(screen.getByLabelText(`Label ${label.name}`)).toHaveAttribute('aria-selected', 'true');
        expect(screen.getByLabelText(`Label ${label.name}`)).toHaveClass(/selected/);
    });

    it('not selected label has no selected styles', async () => {
        const label = {
            id: '123',
            name: 'label',
            color: '#000000',
        };

        render(<LabelListItem label={label} onSelect={vi.fn()} isSelected={false} existingLabels={[]} />);

        expect(screen.getByLabelText(`Label ${label.name}`)).toHaveAttribute('aria-selected', 'false');
        expect(screen.getByLabelText(`Label ${label.name}`)).not.toHaveClass(/selected/);
    });
});
