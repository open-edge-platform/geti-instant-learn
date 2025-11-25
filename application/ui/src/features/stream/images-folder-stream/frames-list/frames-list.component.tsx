/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { RefObject, useCallback, useLayoutEffect, useMemo, useRef } from 'react';

import { type FrameType } from '@geti-prompt/api';
import {
    AriaComponentsListBox,
    DOMRefValue,
    HorizontalLayout,
    HorizontalLayoutOptions,
    ListBoxItem,
    View,
    Virtualizer,
} from '@geti/ui';
import { clsx } from 'clsx';

import styles from './frames-list.module.scss';

interface FrameThumbnailProps {
    frame: FrameType;
    isSelected: boolean;
    onIntersect: (() => void) | undefined;
    rootRef: RefObject<HTMLDivElement | null>;
}

const FrameThumbnail = ({ frame, isSelected, onIntersect, rootRef }: FrameThumbnailProps) => {
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

    return (
        <View
            borderColor={'gray-100'}
            borderYWidth={'thick'}
            borderXWidth={isSelected ? 'thick' : undefined}
            height={'100%'}
            width={'100%'}
            ref={handleRef}
        >
            <View
                UNSAFE_className={clsx(styles.frame, {
                    [styles.selected]: isSelected,
                    [styles.notSelected]: !isSelected,
                })}
                height={'100%'}
                width={'100%'}
            >
                <img
                    alt={'Frame'}
                    src={frame.thumbnail}
                    style={{ objectFit: 'cover', height: '100%', width: '100%', display: 'block' }}
                />
            </View>
        </View>
    );
};

interface FramesListProps {
    activeFrameIndex: number;
    onSetActiveFrame: (index: number) => void;
    frames: FrameType[];
    ref: RefObject<HTMLDivElement | null>;
    fetchNextPage: () => void;
    fetchPreviousPage: () => void;
}

const LAYOUT_OPTIONS = {
    size: 80,
    gap: 0,
    // number of items to render before and after the visible area
    overscan: 5,
} satisfies HorizontalLayoutOptions;

const fulfillWithEmptyFrames = (frames: FrameType[]): FrameType[] => {
    if (frames.length === 0) {
        return frames;
    }

    if (frames[0].index === 0) {
        return frames;
    }

    const emptyFrames: FrameType[] = [];

    for (let i = 0; i < frames[0].index; i++) {
        emptyFrames.push({
            index: i,
            thumbnail: frames[0].thumbnail,
        });
    }

    return [...emptyFrames, ...frames];
};

const useScrollToActiveFrame = (ref: RefObject<HTMLDivElement | null>, activeFrameIndex: number) => {
    useLayoutEffect(() => {
        setTimeout(() => {
            if (ref.current === null) {
                return;
            }
            const itemWidth = LAYOUT_OPTIONS.size + LAYOUT_OPTIONS.gap;
            const activeFrameIndexPosition = activeFrameIndex * itemWidth;

            const isActiveFrameVisible =
                ref.current.scrollLeft <= activeFrameIndexPosition &&
                activeFrameIndexPosition + itemWidth < ref.current.scrollLeft + ref.current.clientWidth;

            if (isActiveFrameVisible) {
                return;
            }

            ref.current.scrollLeft = activeFrameIndexPosition;
        }, 100);

        // Delay to allow Virtualizer to render items and then scroll to the active frame
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
};

const OFFSET_TO_FETCH_NEW_PAGE = 4;

export const FramesList = ({
    activeFrameIndex,
    frames,
    onSetActiveFrame,
    ref,
    fetchNextPage,
    fetchPreviousPage,
}: FramesListProps) => {
    const framesList = useMemo(() => fulfillWithEmptyFrames(frames), [frames]);

    useScrollToActiveFrame(ref, activeFrameIndex);

    return (
        <View height={'100%'} padding={'size-200'} backgroundColor={'gray-100'}>
            <Virtualizer<HorizontalLayoutOptions> layout={HorizontalLayout} layoutOptions={LAYOUT_OPTIONS}>
                <AriaComponentsListBox
                    orientation={'horizontal'}
                    className={styles.framesList}
                    aria-label={'Frames list'}
                    ref={ref}
                >
                    {framesList.map((frame) => {
                        return (
                            <ListBoxItem
                                key={frame.index}
                                className={styles.frameItem}
                                aria-label={`Frame #${frame.index}`}
                                data-isSelected={frame.index === activeFrameIndex}
                                onAction={() => onSetActiveFrame(frame.index)}
                            >
                                <FrameThumbnail
                                    frame={frame}
                                    isSelected={frame.index === activeFrameIndex}
                                    onIntersect={
                                        frame.index === frames[0].index + OFFSET_TO_FETCH_NEW_PAGE
                                            ? fetchPreviousPage
                                            : frame.index === frames[frames.length - 1].index - OFFSET_TO_FETCH_NEW_PAGE
                                              ? fetchNextPage
                                              : undefined
                                    }
                                    rootRef={ref}
                                />
                            </ListBoxItem>
                        );
                    })}
                </AriaComponentsListBox>
            </Virtualizer>
        </View>
    );
};
