/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { RefObject, useLayoutEffect, useRef } from 'react';

import { DOMRefValue, Loading, useUnwrapDOMRef, View } from '@geti/ui';
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
    const ref = useRef<DOMRefValue<HTMLDivElement> | null>(null);
    const unwrappedRef = useUnwrapDOMRef(ref);
    const handleIntersectionRef = useRef(onIntersect);

    useLayoutEffect(() => {
        handleIntersectionRef.current = onIntersect;
    }, [onIntersect]);

    useLayoutEffect(() => {
        if (unwrappedRef.current == null || rootRef.current === null) {
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

        observer.observe(unwrappedRef.current);

        return () => {
            observer.disconnect();
        };
    }, [unwrappedRef, rootRef]);

    return ref;
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
