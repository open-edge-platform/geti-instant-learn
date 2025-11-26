/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { RefObject, useLayoutEffect, useMemo } from 'react';

import { type FrameType } from '@geti-prompt/api';
import {
    AriaComponentsListBox,
    HorizontalLayout,
    HorizontalLayoutOptions,
    ListBoxItem,
    View,
    Virtualizer,
} from '@geti/ui';

import { FrameThumbnail } from './frame-thumbnail.component';

import styles from './frames-list.module.scss';

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
                                data-isselected={frame.index === activeFrameIndex}
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
