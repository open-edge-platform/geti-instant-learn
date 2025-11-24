/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { RefObject } from 'react';

import { type FrameType } from '@geti-prompt/api';
import {
    AriaComponentsListBox,
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
}

const FrameThumbnail = ({ frame, isSelected }: FrameThumbnailProps) => {
    const { thumbnail } = frame;

    return (
        <View
            borderColor={'gray-100'}
            borderYWidth={'thick'}
            borderXWidth={isSelected ? 'thick' : undefined}
            height={'100%'}
            width={'100%'}
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
                    src={thumbnail}
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
    onLoadMore: () => void;
}

const LAYOUT_OPTIONS: HorizontalLayoutOptions = {
    size: 80,
    gap: 0,
    // number of items to render before and after the visible area
    overscan: 5,
};

export const FramesList = ({ activeFrameIndex, frames, onSetActiveFrame, ref, onLoadMore }: FramesListProps) => {
    return (
        <View height={'100%'} padding={'size-200'} backgroundColor={'gray-100'}>
            <Virtualizer<HorizontalLayoutOptions> layout={HorizontalLayout} layoutOptions={LAYOUT_OPTIONS}>
                <AriaComponentsListBox
                    orientation={'horizontal'}
                    className={styles.framesList}
                    aria-label={'Frames list'}
                    ref={ref}
                    onScroll={(event) => {
                        const target = event.target as HTMLDivElement;
                        const fetchNextPageDistance = 500;

                        const { scrollWidth, scrollLeft, clientWidth } = target;

                        const distance = scrollWidth - (clientWidth + scrollLeft);

                        if (distance <= fetchNextPageDistance) {
                            onLoadMore();
                        }
                    }}
                >
                    {frames.map((frame) => (
                        <ListBoxItem
                            key={frame.index}
                            className={styles.frameItem}
                            aria-label={`Frame #${frame.index}`}
                            data-isSelected={frame.index === activeFrameIndex}
                            onAction={() => onSetActiveFrame(frame.index)}
                        >
                            <FrameThumbnail frame={frame} isSelected={frame.index === activeFrameIndex} />
                        </ListBoxItem>
                    ))}
                </AriaComponentsListBox>
            </Virtualizer>
        </View>
    );
};
