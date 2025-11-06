/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { ActionButton, Grid, minmax, View } from '@geti/ui';
import { ChevronLeft, ChevronRight } from '@geti/ui/icons';

import { usePromptMode } from '../../prompts/prompt-modes/prompt-modes.component';
import { CaptureFrameButton } from '../capture-frame-button.component';
import { FramesList, useFrames } from '../frames-list/frames-list.component';
import { Video } from '../video.component';

import styles from './images-folder-stream.module.scss';

export const ImagesFolderStream = () => {
    const promptMode = usePromptMode();
    const frames = useFrames();

    // TODO: replace with actual active frame index
    const [activeFrameIdx, setActiveFrameIdx] = useState(0);

    const isPrevFrameButtonDisabled = activeFrameIdx === 0;
    const isNextFrameButtonDisabled = activeFrameIdx === frames.length - 1;

    const nextFrame = () => {
        setActiveFrameIdx((prev) => prev + 1);
    };

    const prevFrame = () => {
        setActiveFrameIdx((prev) => prev - 1);
    };

    return (
        <Grid
            height={'100%'}
            width={'100%'}
            rows={[minmax(0, '1fr'), 'max-content', '120px']}
            columns={['size-200', 'size-600', minmax(0, '1fr'), 'size-600', 'size-200']}
            areas={[
                'left-gutter prev-btn stream next-btn right-gutter',
                'capture capture capture capture capture',
                'frames frames frames frames frames',
            ]}
            rowGap={'size-200'}
            UNSAFE_style={{
                paddingTop: '48px',
            }}
        >
            <ActionButton
                gridArea={'prev-btn'}
                alignSelf={'center'}
                UNSAFE_className={styles.button}
                isDisabled={isPrevFrameButtonDisabled}
                onPress={prevFrame}
            >
                <ChevronLeft />
            </ActionButton>
            <ActionButton
                gridArea={'next-btn'}
                alignSelf={'center'}
                UNSAFE_className={styles.button}
                isDisabled={isNextFrameButtonDisabled}
                onPress={nextFrame}
            >
                <ChevronRight />
            </ActionButton>
            <View gridArea={'stream'}>
                <Video />
            </View>
            {promptMode === 'visual' && (
                <View gridArea={'capture'} justifySelf={'center'}>
                    <CaptureFrameButton />
                </View>
            )}
            <View gridArea={'frames'}>
                <FramesList activeFrameIndex={activeFrameIdx} frames={frames} />
            </View>
        </Grid>
    );
};
