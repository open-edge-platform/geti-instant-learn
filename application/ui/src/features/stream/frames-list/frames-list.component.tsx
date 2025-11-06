/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { View, VirtualizedHorizontalGrid } from '@geti/ui';
import { clsx } from 'clsx';

import TestImg from '../../../assets/test.webp';

import styles from './frames-list.module.scss';

const useFrames = () => {
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

interface FrameThumbnailProps {
    frame: {
        // it's a base 64 encoded string
        thumbnail: string;
        index: number;
    };
    isSelected: boolean;
}

const FrameThumbnail = ({ frame, isSelected }: FrameThumbnailProps) => {
    const { thumbnail } = frame;

    return (
        <View borderColor={'gray-100'} borderWidth={'thicker'} height={'100%'} width={'100%'}>
            <View
                UNSAFE_className={clsx({
                    [styles.selected]: isSelected,
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

export const FramesList = () => {
    const frames = useFrames();
    // TODO: replace with actual active frame index
    const activeFrameIndex = 0;

    return (
        <View height={'100%'} overflow={'auto hidden'} padding={'size-200'} backgroundColor={'gray-100'}>
            <VirtualizedHorizontalGrid
                items={frames}
                renderItem={(frame) => <FrameThumbnail frame={frame} isSelected={frame.index === activeFrameIndex} />}
                idFormatter={(item) => item.index.toString()}
                textValueFormatter={(item) => item.index.toString()}
                layoutOptions={{ size: 72, gap: 0 }}
                listBoxItemStyles={{
                    height: '100%',
                    width: '100%',
                }}
            />
        </View>
    );
};
