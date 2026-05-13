/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { renderHook } from '@/test-utils';

import { useShowModelLoadingDialog } from './model-loading-dialog.component';

const { mockUseModelLoading, mockUseSpinDelay } = vi.hoisted(() => ({
    mockUseModelLoading: vi.fn(() => false),
    mockUseSpinDelay: vi.fn((value: boolean) => value),
}));

vi.mock('./use-model-loading.hook', () => ({
    useModelLoading: mockUseModelLoading,
    MODEL_STATUS_PATH: '/api/v1/projects/{project_id}/model-status',
    modelStatusQueryKey: vi.fn(),
    startModelStatusProbe: vi.fn(),
    stopModelStatusProbe: vi.fn(),
}));

vi.mock('spin-delay', () => ({
    useSpinDelay: mockUseSpinDelay,
}));

describe('useShowModelLoadingDialog', () => {
    afterEach(() => {
        mockUseModelLoading.mockReturnValue(false);
        mockUseSpinDelay.mockImplementation((value: boolean) => value);
    });

    it('returns false when the model is not loading', () => {
        const { result } = renderHook(() => useShowModelLoadingDialog());

        expect(result.current).toBe(false);
    });

    it('returns true when the model is loading', () => {
        mockUseModelLoading.mockReturnValue(true);

        const { result } = renderHook(() => useShowModelLoadingDialog());

        expect(result.current).toBe(true);
    });

    it('passes correct spin-delay config', () => {
        renderHook(() => useShowModelLoadingDialog());

        expect(mockUseSpinDelay).toHaveBeenCalledWith(false, { delay: 300, minDuration: 400 });
    });
});
