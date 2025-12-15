/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef } from 'react';

import { throttle, type DebouncedFunc } from 'lodash-es';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Callback = (...args: any[]) => void;

const throttleCallback = (callback: Callback, delay: number) => {
    return throttle(callback, delay, {
        leading: true,
        trailing: true,
    });
};

export const useThrottledCallback = (callback: Callback, delay: number): DebouncedFunc<Callback> => {
    const throttledRef = useRef<ReturnType<typeof throttle>>(throttleCallback(callback, delay));

    useEffect(() => {
        throttledRef.current = throttleCallback(callback, delay);

        return () => {
            throttledRef.current?.cancel();
        };
    }, [callback, delay]);

    return throttledRef.current;
};
