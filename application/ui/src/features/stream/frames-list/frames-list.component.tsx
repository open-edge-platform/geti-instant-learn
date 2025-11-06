/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { RefObject } from 'react';

import {
    AriaComponentsListBox,
    HorizontalLayout,
    HorizontalLayoutOptions,
    ListBoxItem,
    View,
    Virtualizer,
} from '@geti/ui';
import { clsx } from 'clsx';

import TestImg from '../../../assets/test.webp';

import styles from './frames-list.module.scss';

export const useFrames = (): Frame[] => {
    // TODO: replace with actual frames
    return [
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
        TestImg,
    ].map((url, idx) => {
        return {
            thumbnail: url,
            index: idx,
        };
    });
};

export interface Frame {
    // it's a base 64 encoded string
    thumbnail: string;
    index: number;
}

interface FrameThumbnailProps {
    frame: Frame;
    isSelected: boolean;
    onActivateFrame: (index: number) => void;
}

const FrameThumbnail = ({ frame, isSelected, onActivateFrame }: FrameThumbnailProps) => {
    const { thumbnail } = frame;

    return (
        <div style={{ height: '100%', width: '100%' }} onClick={() => onActivateFrame(frame.index)}>
            <View
                borderColor={'gray-100'}
                borderYWidth={'thick'}
                borderXWidth={isSelected ? 'thick' : undefined}
                height={'100%'}
                width={'100%'}
            >
                <View
                    UNSAFE_className={clsx({
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
        </div>
    );
};

interface FramesListProps {
    activeFrameIndex: number;
    onSetActiveFrame: (index: number) => void;
    frames: Frame[];
    ref: RefObject<HTMLDivElement | null>;
}

export const FramesList = ({ activeFrameIndex, frames, onSetActiveFrame, ref }: FramesListProps) => {
    return (
        <View height={'100%'} overflow={'hidden'} padding={'size-200'} backgroundColor={'gray-100'}>
            <Virtualizer<HorizontalLayoutOptions> layout={HorizontalLayout} layoutOptions={{ size: 80, gap: 0 }}>
                <AriaComponentsListBox
                    orientation={'horizontal'}
                    style={{ overflowX: 'auto', width: '100%', scrollbarGutter: 'stable' }}
                    aria-label={'Frames list'}
                    ref={ref}
                >
                    {frames.map((frame) => (
                        <ListBoxItem
                            key={frame.index}
                            style={{ height: '100%', width: '100%', outline: 'none' }}
                            aria-label={`Frame #${frame.index}`}
                        >
                            <FrameThumbnail
                                frame={frame}
                                isSelected={frame.index === activeFrameIndex}
                                onActivateFrame={onSetActiveFrame}
                            />
                        </ListBoxItem>
                    ))}
                </AriaComponentsListBox>
            </Virtualizer>
        </View>
    );
};
