/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useRef, useState } from 'react';

import { useEventListener } from '@geti-prompt/hooks';
import { ActionButton, Grid, minmax, View } from '@geti/ui';
import { ChevronLeft, ChevronRight } from '@geti/ui/icons';

import { usePromptMode } from '../../prompts/prompt-modes/prompt-modes.component';
import { CaptureFrameButton } from '../capture-frame-button.component';
import { FramesList, useFrames, type Frame } from '../frames-list/frames-list.component';
import { Video } from '../video.component';

import styles from './images-folder-stream.module.scss';

const useActiveFrameSelection = (frames: Frame[]) => {
    // TODO: replace with actual active frame index
    const [activeFrameIdx, setActiveFrameIdx] = useState(0);
    const framesRef = useRef<HTMLDivElement>(null);

    useEventListener('keydown', (event) => {
        if (event.key === 'ArrowLeft') {
            prevFrame();
        } else if (event.key === 'ArrowRight') {
            nextFrame();
        }
    });

    const scrollFrameIntoView = (frameIdx: number) => {
        const framesListRect = framesRef.current?.getBoundingClientRect();
        // Note: we don't want to create an array of refs for each frame, so we use querySelector
        const frameElement = document.querySelector(`[aria-label="Frame #${frameIdx}"]`);

        if (frameElement === null || framesListRect === undefined) {
            return;
        }

        const frameRect = frameElement.getBoundingClientRect();

        const shouldScroll = frameRect.left < framesListRect.left || frameRect.right > framesListRect.right;

        if (shouldScroll) {
            frameElement.scrollIntoView({
                behavior: 'smooth',
            });
        }
    };

    const activateFrame = (frameIdx: number) => {
        setActiveFrameIdx(frameIdx);
        scrollFrameIntoView(frameIdx);
    };

    const nextFrame = () => {
        const nextFrameIdx = activeFrameIdx + 1;

        if (nextFrameIdx >= frames.length) {
            return;
        }

        activateFrame(nextFrameIdx);
    };

    const prevFrame = () => {
        const prevFrameIdx = activeFrameIdx - 1;
        if (prevFrameIdx < 0) {
            return;
        }
        activateFrame(prevFrameIdx);
    };

    return {
        framesRef,
        activeFrameIdx,
        activateFrame,
        nextFrame,
        prevFrame,
    };
};

export const ImagesFolderStream = () => {
    const promptMode = usePromptMode();
    const frames = useFrames();
    const { activeFrameIdx, activateFrame, nextFrame, prevFrame, framesRef } = useActiveFrameSelection(frames);

    const isPrevFrameButtonDisabled = activeFrameIdx === 0;
    const isNextFrameButtonDisabled = activeFrameIdx === frames.length - 1;

    return (
        <Grid
            height={'100%'}
            width={'100%'}
            rows={[minmax(0, '1fr'), 'max-content', 'size-1600']}
            columns={['size-200', 'size-600', minmax(0, '1fr'), 'size-600', 'size-200']}
            areas={[
                'left-gutter prev-btn stream next-btn right-gutter',
                'capture capture capture capture capture',
                'frames frames frames frames frames',
            ]}
            gap={'size-200'}
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
                <FramesList
                    ref={framesRef}
                    activeFrameIndex={activeFrameIdx}
                    onSetActiveFrame={activateFrame}
                    frames={frames}
                />
            </View>
        </Grid>
    );
};
