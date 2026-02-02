/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { getMockedLabel, render } from '@/test-utils';
import { fireEvent, screen } from '@testing-library/react';
import { ZoomState } from 'src/components/zoom/types';
import { useZoom } from 'src/components/zoom/zoom.provider';

import { Label } from '../types';
import { AnnotationLabels } from './annotation-labels.component';

const mockZoom: ZoomState = {
    scale: 1,
    maxZoomIn: 10,
    hasAnimation: false,
    translate: { x: 0, y: 0 },
    initialCoordinates: { x: 0, y: 0, scale: 1 },
};

vi.mock('src/components/zoom/zoom.provider', () => ({
    useZoom: vi.fn(),
}));

const renderAnnotationLabels = (labels: Label[], onRemove: (id: string) => void) => {
    render(
        <svg>
            <AnnotationLabels labels={labels} onRemove={onRemove} />
        </svg>
    );
};

describe('AnnotationLabels', () => {
    const mockOnRemove = vi.fn();

    beforeEach(() => {
        vi.mocked(useZoom).mockReturnValue(mockZoom);
    });

    afterEach(() => {
        mockOnRemove.mockClear();
        vi.clearAllMocks();
    });

    it('renders placeholder when no labels provided', () => {
        renderAnnotationLabels([], mockOnRemove);

        expect(screen.getByText('No label')).toBeInTheDocument();
    });

    it('renders single label with name and color', () => {
        const label = getMockedLabel({ name: 'Person', color: '#FF0000' });

        renderAnnotationLabels([label], mockOnRemove);

        expect(screen.getByText(label.name)).toBeInTheDocument();

        const rect = screen.getByLabelText(`label ${label.name} background`);
        expect(rect).toHaveAttribute('fill', label.color);
    });

    it('renders multiple labels', () => {
        const labels: Label[] = [
            getMockedLabel({ id: '1', name: 'Person', color: '#FF0000' }),
            getMockedLabel({ id: '2', name: 'Car', color: '#00FF00' }),
        ];

        renderAnnotationLabels(labels, mockOnRemove);

        expect(screen.getByText(labels[0].name)).toBeInTheDocument();
        expect(screen.getByText(labels[1].name)).toBeInTheDocument();
    });

    it('calls onRemove when close button clicked', () => {
        const label = getMockedLabel({ id: 'label-1', name: 'Person' });

        renderAnnotationLabels([label], mockOnRemove);

        const closeButton = screen.getByLabelText(`Remove ${label.name}`);
        fireEvent.pointerDown(closeButton);

        expect(mockOnRemove).toHaveBeenCalledTimes(1);
        expect(mockOnRemove).toHaveBeenCalledWith(label.id);
    });

    it('adjusts sizes based on zoom scale', () => {
        const newMockZoom: ZoomState = { ...mockZoom, scale: 2 };

        vi.mocked(useZoom).mockReturnValue(newMockZoom);

        const label = getMockedLabel({ name: 'Person' });

        renderAnnotationLabels([label], mockOnRemove);

        const text = screen.getByLabelText(`label ${label.name}`);

        // Font size should be 14 / 2 = 7
        expect(text).toHaveAttribute('font-size', '7');
    });

    it('prevents event propagation on close button click', () => {
        const label = getMockedLabel({ id: 'label-1', name: 'Person' });
        const mockParentHandler = vi.fn();

        render(
            <svg onPointerDown={mockParentHandler}>
                <AnnotationLabels labels={[label]} onRemove={mockOnRemove} />
            </svg>
        );

        const closeButton = screen.getByLabelText(`Remove ${label.name}`);
        fireEvent.pointerDown(closeButton);

        expect(mockOnRemove).toHaveBeenCalled();
        // Parent handler should not be called due to stopPropagation
        expect(mockParentHandler).not.toHaveBeenCalled();
    });

    it('renders labels with correct positioning (no overlap)', () => {
        const labels: Label[] = [
            getMockedLabel({ id: '1', name: 'A', color: '#FF0000' }),
            getMockedLabel({ id: '2', name: 'B', color: '#00FF00' }),
        ];

        renderAnnotationLabels(labels, mockOnRemove);

        const firstRect = screen.getByLabelText(`label ${labels[0].name} background`);
        const secondRect = screen.getByLabelText(`label ${labels[1].name} background`);

        const firstX = parseFloat(firstRect.getAttribute('x') || '0');
        const secondX = parseFloat(secondRect.getAttribute('x') || '0');

        // Second label should be positioned after the first
        expect(secondX).toBeGreaterThan(firstX);
    });
});
