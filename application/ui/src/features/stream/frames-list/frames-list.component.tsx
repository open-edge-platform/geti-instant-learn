/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback } from 'react';

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

import TestImg from '../../../assets/test.webp';

import styles from './frames-list.module.scss';

export const useFrames = () => {
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

interface Frame {
    // it's a base 64 encoded string
    thumbnail: string;
    index: number;
}

interface FrameThumbnailProps {
    frame: Frame;
    isSelected: boolean;
}

const FrameThumbnail = ({ frame, isSelected }: FrameThumbnailProps) => {
    const { thumbnail } = frame;

    const refHandler = useCallback(
        (ref: DOMRefValue<HTMLElement> | null) => {
            const element = ref?.UNSAFE_getDOMNode();
            if (element == null) {
                return;
            }

            if (!isSelected) {
                return;
            }

            element.scrollIntoView({
                behavior: 'smooth',
            });
        },
        [isSelected]
    );

    return (
        <View ref={refHandler} borderColor={'gray-100'} borderWidth={'thicker'} height={'100%'} width={'100%'}>
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
    );
};

interface FramesListProps {
    activeFrameIndex: number;
    frames: Frame[];
}

export const FramesList = ({ activeFrameIndex, frames }: FramesListProps) => {
    return (
        <View height={'100%'} overflow={'hidden'} padding={'size-200'} backgroundColor={'gray-100'}>
            <Virtualizer<HorizontalLayoutOptions> layout={HorizontalLayout} layoutOptions={{ size: 80, gap: 0 }}>
                <AriaComponentsListBox orientation={'horizontal'} style={{ overflowX: 'auto', width: '100%' }}>
                    {frames.map((frame) => (
                        <ListBoxItem key={frame.index} style={{ height: '100%', width: '100%' }}>
                            <FrameThumbnail frame={frame} isSelected={frame.index === activeFrameIndex} />
                        </ListBoxItem>
                    ))}
                </AriaComponentsListBox>
            </Virtualizer>
        </View>
    );
};
