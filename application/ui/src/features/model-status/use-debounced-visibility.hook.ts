/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from 'react';

/**
 * Anti-flicker hook: delays showing by `enterDelay` ms and ensures the
 * visible state persists for at least `minVisible` ms once shown.
 */
export const useDebouncedVisibility = (active: boolean, enterDelay = 200, minVisible = 400): boolean => {
    const [visible, setVisible] = useState(false);
    const shownAtRef = useRef<number>(0);
    const enterTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
    const exitTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

    useEffect(() => {
        if (active) {
            // Cancel any pending exit.
            clearTimeout(exitTimerRef.current);

            // Debounce enter.
            enterTimerRef.current = setTimeout(() => {
                shownAtRef.current = Date.now();
                setVisible(true);
            }, enterDelay);
        } else {
            // Cancel any pending enter.
            clearTimeout(enterTimerRef.current);

            if (!visible) return;

            const elapsed = Date.now() - shownAtRef.current;
            const remaining = minVisible - elapsed;

            if (remaining > 0) {
                exitTimerRef.current = setTimeout(() => setVisible(false), remaining);
            } else {
                setVisible(false);
            }
        }

        return () => {
            clearTimeout(enterTimerRef.current);
            clearTimeout(exitTimerRef.current);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [active]);

    return visible;
};
