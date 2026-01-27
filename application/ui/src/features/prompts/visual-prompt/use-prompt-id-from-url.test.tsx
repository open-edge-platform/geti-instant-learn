/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { renderHook } from '@/test-utils';
import { act } from '@testing-library/react';

import { usePromptIdFromUrl } from './use-prompt-id-from-url';

describe('usePromptIdFromUrl', () => {
    it('returns null when promptId is not in URL', () => {
        const { result } = renderHook(() => usePromptIdFromUrl(), { route: '/', path: '' });

        expect(result.current.promptId).toBeNull();
    });

    it('handles empty string promptId', async () => {
        const { result } = renderHook(() => usePromptIdFromUrl(), { route: '/', path: '' });

        act(() => {
            result.current.setPromptId('');
        });

        expect(result.current.promptId).toBe('');
    });

    it('returns promptId when it exists in URL', () => {
        const { result } = renderHook(() => usePromptIdFromUrl(), {
            route: '/?promptId=sir-prompt-a-lot',
            path: '',
        });

        expect(result.current.promptId).toBe('sir-prompt-a-lot');
    });

    it('sets promptId in URL when setPromptId is called', async () => {
        const { result } = renderHook(() => usePromptIdFromUrl(), { route: '/', path: '' });

        expect(result.current.promptId).toBeNull();

        act(() => {
            result.current.setPromptId('mega-prompt');
        });

        expect(result.current.promptId).toBe('mega-prompt');
    });

    it('updates promptId in URL when it already exists', async () => {
        const { result } = renderHook(() => usePromptIdFromUrl(), {
            route: '/?promptId=previous-prompt',
            path: '',
        });

        expect(result.current.promptId).toBe('previous-prompt');

        act(() => {
            result.current.setPromptId('current-prompt');
        });

        expect(result.current.promptId).toBe('current-prompt');
    });

    it('removes promptId from URL when setPromptId is called with null', async () => {
        const { result } = renderHook(() => usePromptIdFromUrl(), {
            route: '/?promptId=to-be-removed',
            path: '',
        });

        expect(result.current.promptId).toBe('to-be-removed');

        act(() => {
            result.current.setPromptId(null);
        });

        expect(result.current.promptId).toBeNull();
    });
});
