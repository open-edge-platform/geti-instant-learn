/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from 'react';

import { LabelType } from '@geti-prompt/api';
import { Point } from '@geti/smart-tools/types';
import { ActionButton, Flex, Overlay, View } from '@geti/ui';
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

interface CreateLabelProps {
    point: Point | undefined;
    onClose: () => void;
    onSuccess: (label: LabelType) => void;
    existingLabels: LabelType[];
}

export const CreateLabel = ({ point, onClose, onSuccess, existingLabels }: CreateLabelProps) => {
    const nodeRef = useRef(null);

    if (point === undefined) return null;

    return (
        <Overlay isOpen nodeRef={nodeRef}>
            <div
                onPointerDown={(event) => {
                    event.stopPropagation();
                }}
                onPointerMove={(event) => event.stopPropagation()}
                style={{
                    position: 'absolute',
                    left: point?.x,
                    top: point?.y,
                    transform: 'translate(-50%, -50%)',
                }}
            >
                <View
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
