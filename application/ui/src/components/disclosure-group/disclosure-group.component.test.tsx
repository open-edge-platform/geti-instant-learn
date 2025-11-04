/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen } from '@testing-library/react';

import { DisclosureGroup } from './disclosure-group.component';

describe('DisclosureGroup', () => {
    const mockItems = [
        {
            value: 'item1',
            label: 'Item 1',
            icon: <span aria-label='icon-1'>Icon1</span>,
            content: <div>Content 1</div>,
        },
        {
            value: 'item2',
            label: 'Item 2',
            icon: <span aria-label='icon-2'>Icon2</span>,
            content: <div>Content 2</div>,
        },
        {
            value: 'item3',
            label: 'Item 3',
            icon: <span aria-label='icon-3'>Icon3</span>,
        },
    ];

    it('renders all items with labels and icons', () => {
        render(<DisclosureGroup items={mockItems} value={null} />);

        expect(screen.getByText('Item 1')).toBeInTheDocument();
        expect(screen.getByText('Item 2')).toBeInTheDocument();
        expect(screen.getByText('Item 3')).toBeInTheDocument();
        expect(screen.getByLabelText('icon-1')).toBeInTheDocument();
        expect(screen.getByLabelText('icon-2')).toBeInTheDocument();
        expect(screen.getByLabelText('icon-3')).toBeInTheDocument();
    });

    it('expands item matching the value prop', () => {
        render(<DisclosureGroup items={mockItems} value='item1' />);

        expect(screen.getByText('Content 1')).toBeInTheDocument();
        expect(screen.queryByText('Content 2')).not.toBeInTheDocument();
    });

    it('calls onChange when item is clicked', () => {
        const onChange = vi.fn();
        render(<DisclosureGroup items={mockItems} value={null} onChange={onChange} />);

        const itemButton = screen.getByText('Item 1').closest('button');
        expect(itemButton).not.toBeNull();

        if (itemButton) {
            fireEvent.click(itemButton);
        }

        expect(onChange).toHaveBeenCalledWith('item1');
    });

    it('expands clicked item and shows its content', () => {
        render(<DisclosureGroup items={mockItems} value={null} />);

        const itemButton = screen.getByText('Item 2').closest('button');
        expect(itemButton).not.toBeNull();

        if (itemButton) {
            fireEvent.click(itemButton);
        }

        expect(screen.getByText('Content 2')).toBeInTheDocument();
    });

    it('changes expanded state when clicking different items', () => {
        const onChange = vi.fn();
        render(<DisclosureGroup items={mockItems} value='item1' onChange={onChange} />);

        expect(screen.getByText('Content 1')).toBeInTheDocument();

        const itemButton = screen.getByText('Item 2').closest('button');
        expect(itemButton).not.toBeNull();

        if (itemButton) {
            fireEvent.click(itemButton);
        }

        expect(onChange).toHaveBeenCalledWith('item2');
        expect(screen.getByText('Content 2')).toBeInTheDocument();
    });

    it('collapses item when clicking it again', () => {
        render(<DisclosureGroup items={mockItems} value='item1' />);

        expect(screen.getByText('Content 1')).toBeInTheDocument();

        const itemButton = screen.getByText('Item 1').closest('button');
        expect(itemButton).not.toBeNull();

        if (itemButton) {
            fireEvent.click(itemButton);
        }

        expect(screen.queryByText('Content 1')).not.toBeInTheDocument();
    });
});
