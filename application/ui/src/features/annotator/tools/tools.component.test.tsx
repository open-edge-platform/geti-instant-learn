/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen } from '@testing-library/react';

import { ToolConfig } from './interface';
import { Tools } from './tools.component';

describe('Tools', () => {
    const mockTools: ToolConfig[] = [{ type: 'sam', icon: () => <svg data-testid='sam-icon' /> }];

    it('does not render if there are no tools', () => {
        render(<Tools tools={[]} activeTool={null} setActiveTool={() => {}} />);

        expect(screen.queryByRole('button')).toBeNull();
    });

    it('renders tools correctly', () => {
        render(<Tools tools={mockTools} activeTool={'sam'} setActiveTool={() => {}} />);

        const samButton = screen.getByLabelText('Select sam Tool');

        expect(samButton).toBeInTheDocument();
        expect(screen.getByTestId('sam-icon')).toBeInTheDocument();
    });

    it('sets the active tool on button press', () => {
        const setActiveToolMock = vi.fn();

        render(<Tools tools={mockTools} activeTool={null} setActiveTool={setActiveToolMock} />);

        const samButton = screen.getByLabelText('Select sam Tool');
        fireEvent.click(samButton);

        expect(setActiveToolMock).toHaveBeenCalledWith('sam');
    });
});
