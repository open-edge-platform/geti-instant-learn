/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { RefObject, useCallback, useLayoutEffect, useRef } from 'react';

import { DOMRefValue, Loading, View } from '@geti/ui';
import { clsx } from 'clsx';

import { type FrameType } from '../api/interface';

import styles from './frames-list.module.scss';

interface FrameThumbnailProps {
    frame: FrameType;
    isSelected: boolean;
    onIntersect: (() => void) | undefined;
    rootRef: RefObject<HTMLDivElement | null>;
}

const useObserveThumbnail = (rootRef: RefObject<HTMLDivElement | null>, onIntersect: (() => void) | undefined) => {
    const handleIntersectionRef = useRef(onIntersect);

    useLayoutEffect(() => {
        handleIntersectionRef.current = onIntersect;
    }, [onIntersect]);

    const handleRef = useCallback(
        (domRefValue: DOMRefValue<HTMLElement> | null) => {
            const ref = domRefValue?.UNSAFE_getDOMNode();

            if (ref == null || rootRef.current === null) {
                return;
            }

            if (handleIntersectionRef.current === undefined) {
                return;
            }

            const observer = new IntersectionObserver(
                (entries) => {
                    if (entries.length === 0) {
                        return;
                    }

                    if (entries[0].isIntersecting) {
                        handleIntersectionRef.current?.();
                    }
                },
                {
                    threshold: 0.01,
                    rootMargin: '200px',
                    root: rootRef.current,
                }
            );

            observer.observe(ref);

            return () => {
                observer.disconnect();
            };
        },
        [rootRef]
    );

    return handleRef;
};

const FrameThumbnailPlaceholder = () => {
    return <Loading mode={'inline'} size={'S'} UNSAFE_style={{ height: '100%', width: '100%' }} />;
};

export const FrameThumbnail = ({ frame, isSelected, onIntersect, rootRef }: FrameThumbnailProps) => {
    const ref = useObserveThumbnail(rootRef, onIntersect);

    return (
        <View
            borderColor={'gray-100'}
            borderYWidth={'thick'}
            borderXWidth={isSelected ? 'thick' : undefined}
            height={'100%'}
            width={'100%'}
            ref={ref}
        >
            <View
                UNSAFE_className={clsx(styles.frame, {
                    [styles.selected]: isSelected,
                    [styles.notSelected]: !isSelected,
                })}
                height={'100%'}
                width={'100%'}
            >
                {frame.thumbnail === null ? (
                    <FrameThumbnailPlaceholder />
                ) : (
                    <img alt={'Frame'} src={frame.thumbnail} className={styles.frameImg} />
                )}
            </View>
        </View>
    );
};
