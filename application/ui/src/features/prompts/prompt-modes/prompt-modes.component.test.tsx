/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen } from '@testing-library/react';

import { PromptModes, usePromptMode } from './prompt-modes.component';

describe('PromptModes', () => {
    it('renders prompt mode toggle buttons', () => {
        render(<PromptModes />);

        expect(screen.getByText('Prompt Mode')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Visual Prompt' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Text Prompt' })).toBeInTheDocument();
    });

    it('defaults to visual mode when no mode is set', () => {
        render(<PromptModes />);

        const visualButton = screen.getByRole('button', { name: 'Visual Prompt' });
        expect(visualButton).toHaveAttribute('aria-pressed', 'true');
    });

    it('shows visual mode as selected when mode=visual in URL', () => {
        render(<PromptModes />, { route: '/prompts?mode=visual', path: '/prompts' });

        const visualButton = screen.getByRole('button', { name: 'Visual Prompt' });
        expect(visualButton).toHaveAttribute('aria-pressed', 'true');
    });

    it('shows text mode as selected when mode=text in URL', () => {
        render(<PromptModes />, { route: '/prompts?mode=text', path: '/prompts' });

        const textButton = screen.getByRole('button', { name: 'Text Prompt' });
        expect(textButton).toHaveAttribute('aria-pressed', 'true');
    });

    it('updates URL when switching from visual to text mode', async () => {
        render(<PromptModes />, { route: '/prompts?mode=visual', path: '/prompts' });

        const visualButton = screen.getByRole('button', { name: 'Visual Prompt' });
        expect(visualButton).toHaveAttribute('aria-pressed', 'true');

        const textButton = screen.getByRole('button', { name: 'Text Prompt' });
        fireEvent.click(textButton);

        expect(textButton).toHaveAttribute('aria-pressed', 'true');
        expect(visualButton).toHaveAttribute('aria-pressed', 'false');
    });

    it('updates URL when switching from text to visual mode', async () => {
        render(<PromptModes />, { route: '/prompts?mode=text', path: '/prompts' });

        const textButton = screen.getByRole('button', { name: 'Text Prompt' });
        expect(textButton).toHaveAttribute('aria-pressed', 'true');

        const visualButton = screen.getByRole('button', { name: 'Visual Prompt' });
        fireEvent.click(visualButton);

        expect(visualButton).toHaveAttribute('aria-pressed', 'true');
        expect(textButton).toHaveAttribute('aria-pressed', 'false');
    });

    it('sets visual mode in URL on initial render when no mode is present', () => {
        render(<PromptModes />);

        const visualButton = screen.getByRole('button', { name: 'Visual Prompt' });
        expect(visualButton).toHaveAttribute('aria-pressed', 'true');
    });
});

describe('usePromptMode', () => {
    const TestComponent = () => {
        const mode = usePromptMode();

        return <div aria-label='mode'>{mode}</div>;
    };

    it('returns visual as default mode', () => {
        render(<TestComponent />, { route: '/prompts', path: '/prompts' });

        expect(screen.getByLabelText('mode')).toHaveTextContent('visual');
    });

    it('returns visual when mode=visual in URL', () => {
        render(<TestComponent />, { route: '/prompts?mode=visual', path: '/prompts' });

        expect(screen.getByLabelText('mode')).toHaveTextContent('visual');
    });

    it('returns text when mode=text in URL', () => {
        render(<TestComponent />, { route: '/prompts?mode=text', path: '/prompts' });

        expect(screen.getByLabelText('mode')).toHaveTextContent('text');
    });
});
