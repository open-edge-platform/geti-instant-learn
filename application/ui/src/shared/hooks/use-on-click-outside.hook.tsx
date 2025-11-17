/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { RefObject, useEffect, useRef } from 'react';

import { TextFieldRef } from '@geti/ui';

export const useOnOutsideClick = (textFieldRef: RefObject<TextFieldRef | null>, onClickOutside: () => void) => {
    const resetProjectInEditionRef = useRef(onClickOutside);

    useEffect(() => {
        resetProjectInEditionRef.current = onClickOutside;
    }, [onClickOutside]);

    useEffect(() => {
        const abortController = new AbortController();

        document.addEventListener(
            'click',
            (event) => {
                if (!textFieldRef.current?.UNSAFE_getDOMNode()?.contains(event.target as Node)) {
                    resetProjectInEditionRef.current();
                }
            },
            { signal: abortController.signal }
        );
        return () => {
            abortController.abort();
        };
    }, [textFieldRef]);
};
