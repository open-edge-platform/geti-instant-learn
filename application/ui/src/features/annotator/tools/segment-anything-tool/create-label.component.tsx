/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from 'react';

import { LabelType } from '@geti-prompt/api';
import { Point } from '@geti/smart-tools/types';
import { ActionButton, DOMRefValue, Flex, Overlay, View } from '@geti/ui';
import { Close } from '@geti/ui/icons';

import { CreateLabelForm } from '../../../prompts/visual-prompt/labels-management/add-label/create-label-form.component';

export const useCreateLabelFormPosition = () => {
    const [point, setPoint] = useState<Point | undefined>(undefined);

    useEffect(() => {
        if (point === undefined) return;

        const abortController = new AbortController();

        const initialWindowSize = {
            width: window.innerWidth,
            height: window.innerHeight,
            x: point.x,
            y: point.y,
        };

        window.addEventListener(
            'resize',
            () => {
                const xDiff = window.innerWidth - initialWindowSize.width;
                const yDiff = window.innerHeight - initialWindowSize.height;

                const newX = initialWindowSize.x + xDiff;
                const newY = initialWindowSize.y + yDiff;

                if (newX === point.x && newY === point.y) return;

                setPoint({
                    x: initialWindowSize.x + xDiff,
                    y: initialWindowSize.y + yDiff,
                });
            },
            {
                signal: abortController.signal,
            }
        );

        return () => {
            abortController.abort();
        };
    }, [point]);

    return [point, setPoint] as const;
};

const useCloseOnOutsideClick = (onClose: () => void) => {
    const labelContainerRef = useRef<DOMRefValue<HTMLDivElement>>(null);
    const onClickOutsideRef = useRef(onClose);

    useEffect(() => {
        onClickOutsideRef.current = onClose;
    }, [onClose]);

    useEffect(() => {
        const abortController = new AbortController();

        document.addEventListener(
            'click',
            (event) => {
                const node = labelContainerRef.current;

                if (node === null) return;

                const nodeBoundingBox = node.UNSAFE_getDOMNode()?.getBoundingClientRect();

                if (nodeBoundingBox === undefined) return;

                if (
                    event.clientY < nodeBoundingBox.top ||
                    event.clientY > nodeBoundingBox.bottom ||
                    event.clientX < nodeBoundingBox.left ||
                    event.clientX > nodeBoundingBox.right
                ) {
                    onClickOutsideRef.current();

                    return;
                }
            },
            { signal: abortController.signal }
        );
        return () => {
            abortController.abort();
        };
    }, [labelContainerRef]);

    return labelContainerRef;
};

interface CreateLabelProps {
    onSuccess: (label: LabelType) => void;
    existingLabels: LabelType[];
    mousePosition: Point | undefined;
    onClose: () => void;
}

export const CreateLabel = ({ onSuccess, existingLabels, onClose, mousePosition }: CreateLabelProps) => {
    const nodeRef = useRef(null);

    const labelContainerRef = useCloseOnOutsideClick(onClose);

    if (mousePosition === undefined) return null;

    return (
        <Overlay isOpen nodeRef={nodeRef}>
            <div
                style={{
                    position: 'absolute',
                    left: mousePosition?.x,
                    top: mousePosition?.y,
                    transform: 'translate(-50%, -50%)',
                }}
            >
                <View
                    ref={labelContainerRef}
                    backgroundColor={'gray-50'}
                    padding={'size-200'}
                    height={'100%'}
                    borderRadius={'regular'}
                    borderWidth={'thin'}
                    borderColor={'gray-400'}
                >
                    <Flex justifyContent={'space-between'} gap={'size-100'}>
                        <CreateLabelForm onClose={onClose} onSuccess={onSuccess} existingLabels={existingLabels} />
                        <ActionButton isQuiet onPress={onClose}>
                            <Close />
                        </ActionButton>
                    </Flex>
                </View>
            </div>
        </Overlay>
    );
};
