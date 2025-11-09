/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { createRef, PointerEvent } from 'react';

import { act, fireEvent, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useContainerSize } from './use-container-size.hook';
import { usePanning } from './use-panning.hook';
import { useWheelPanning } from './use-wheel-panning.hook';

describe('useContainerSize', () => {
    it('returns initial size of 100x100', () => {
        const ref = createRef<HTMLDivElement>();
        const { result } = renderHook(() => useContainerSize(ref));

        expect(result.current).toEqual({ width: 100, height: 100 });
    });

    it('returns container size when ref has an element', () => {
        const div = document.createElement('div');
        Object.defineProperty(div, 'clientWidth', { value: 500, configurable: true });
        Object.defineProperty(div, 'clientHeight', { value: 300, configurable: true });

        const ref = { current: div };
        const { result } = renderHook(() => useContainerSize(ref));

        expect(result.current).toEqual({ width: 100, height: 100 });
    });

    it('does not update size if ref is null', () => {
        const ref = { current: null };
        const { result } = renderHook(() => useContainerSize(ref));

        expect(result.current).toEqual({ width: 100, height: 100 });
    });
});

describe('usePanning', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('returns initial isPanning state as false', () => {
        const { result } = renderHook(() => usePanning());

        expect(result.current.isPanning).toBe(false);
    });

    it('sets isPanning to true when Space key is pressed', () => {
        const { result } = renderHook(() => usePanning());

        act(() => {
            fireEvent.keyDown(window, { code: 'Space' });
        });

        expect(result.current.isPanning).toBe(true);
    });

    it('sets isPanning to false when Space key is released', () => {
        const { result } = renderHook(() => usePanning());

        act(() => {
            fireEvent.keyDown(window, { code: 'Space' });
        });

        expect(result.current.isPanning).toBe(true);

        act(() => {
            fireEvent.keyUp(window, { code: 'Space' });
        });

        expect(result.current.isPanning).toBe(false);
    });
});

describe('useWheelPanning', () => {
    let setIsPanning: ReturnType<typeof vi.fn>;

    beforeEach(() => {
        setIsPanning = vi.fn();
    });

    it('returns initial isGrabbing state as false', () => {
        const { result } = renderHook(() => useWheelPanning(setIsPanning));

        expect(result.current.isGrabbing).toBe(false);
    });

    it('sets isGrabbing to true on pointer down', () => {
        const { result } = renderHook(() => useWheelPanning(setIsPanning));

        act(() => {
            const event = {
                clientX: 100,
                clientY: 100,
                button: 1, // Middle mouse button
            } as PointerEvent<HTMLDivElement>;
            result.current.onPointerDown(event);
        });

        expect(result.current.isGrabbing).toBe(true);
    });

    it('calls setIsPanning(true) when middle mouse button is pressed', () => {
        const { result } = renderHook(() => useWheelPanning(setIsPanning));

        act(() => {
            const event = {
                clientX: 100,
                clientY: 100,
                button: 1, // Middle mouse button
            } as PointerEvent<HTMLDivElement>;
            result.current.onPointerDown(event);
        });

        expect(setIsPanning).toHaveBeenCalledWith(true);
    });

    it('resets state on pointer up', () => {
        const { result } = renderHook(() => useWheelPanning(setIsPanning));

        act(() => {
            const event = {
                clientX: 100,
                clientY: 100,
                button: 1,
            } as PointerEvent<HTMLDivElement>;
            result.current.onPointerDown(event);
        });

        expect(result.current.isGrabbing).toBe(true);

        act(() => {
            result.current.onPointerUp();
        });

        expect(result.current.isGrabbing).toBe(false);
        expect(setIsPanning).toHaveBeenCalledWith(false);
    });

    it('resets state on mouse leave', () => {
        const { result } = renderHook(() => useWheelPanning(setIsPanning));

        act(() => {
            const event = {
                clientX: 100,
                clientY: 100,
                button: 1,
            } as PointerEvent<HTMLDivElement>;
            result.current.onPointerDown(event);
        });

        act(() => {
            result.current.onMouseLeave();
        });

        expect(result.current.isGrabbing).toBe(false);
        expect(setIsPanning).toHaveBeenCalledWith(false);
    });

    it('calls callback with delta when pointer moves with middle button', () => {
        const { result } = renderHook(() => useWheelPanning(setIsPanning));
        const callback = vi.fn();

        act(() => {
            const downEvent = {
                clientX: 100,
                clientY: 100,
                button: 1,
            } as PointerEvent<HTMLDivElement>;
            result.current.onPointerDown(downEvent);
        });

        act(() => {
            const moveEvent = {
                clientX: 150,
                clientY: 120,
                button: 1,
            } as PointerEvent<HTMLDivElement>;
            result.current.onPointerMove(callback)(moveEvent);
        });

        expect(callback).toHaveBeenCalledWith({ x: 50, y: 20 });
    });

    it('does not call callback when pointer moves without middle button pressed', () => {
        const { result } = renderHook(() => useWheelPanning(setIsPanning));
        const callback = vi.fn();

        act(() => {
            const moveEvent = {
                clientX: 150,
                clientY: 120,
                button: 0, // Left button
            } as PointerEvent<HTMLDivElement>;
            result.current.onPointerMove(callback)(moveEvent);
        });

        expect(callback).not.toHaveBeenCalled();
    });
});
