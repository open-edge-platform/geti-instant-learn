/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { RefObject } from 'react';

import { type FrameType } from '@geti-prompt/api';
import {
    AriaComponentsListBox,
    Collection,
    HorizontalLayout,
    HorizontalLayoutOptions,
    ListBoxItem,
    ListBoxLoadMoreItem,
    Loading,
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
                    src={`data:image/jpeg;base64,${thumbnail}`}
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
    isLoadingMore: boolean;
}

const LAYOUT_OPTIONS: HorizontalLayoutOptions = {
    size: 80,
    gap: 0,
    // number of items to render before and after the visible area
    overscan: 5,
};

export const FramesList = ({
    activeFrameIndex,
    frames,
    onSetActiveFrame,
    ref,
    onLoadMore,
    isLoadingMore,
}: FramesListProps) => {
    return (
        <View height={'100%'} padding={'size-200'} backgroundColor={'gray-100'}>
            <Virtualizer<HorizontalLayoutOptions> layout={HorizontalLayout} layoutOptions={LAYOUT_OPTIONS}>
                <AriaComponentsListBox
                    orientation={'horizontal'}
                    className={styles.framesList}
                    aria-label={'Frames list'}
                    items={frames}
                    ref={ref}
                >
                    <Collection items={frames}>
                        {(frame) => (
                            <ListBoxItem
                                id={frame.index}
                                key={frame.index}
                                className={styles.frameItem}
                                aria-label={`Frame #${frame.index}`}
                                data-isSelected={frame.index === activeFrameIndex}
                                onAction={() => onSetActiveFrame(frame.index)}
                            >
                                <FrameThumbnail frame={frame} isSelected={frame.index === activeFrameIndex} />
                            </ListBoxItem>
                        )}
                    </Collection>

                    <ListBoxLoadMoreItem
                        onLoadMore={onLoadMore}
                        isLoading={isLoadingMore}
                        aria-label={'Load more frames'}
                        data-testid={'load-more-frames'}
                    >
                        <Loading mode={'inline'} />
                    </ListBoxLoadMoreItem>
                </AriaComponentsListBox>
            </Virtualizer>
        </View>
    );
};
