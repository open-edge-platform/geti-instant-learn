/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useRef } from 'react';

import { FrameType } from '@geti-prompt/api';
import { useEventListener } from '@geti-prompt/hooks';
import { ActionButton, dimensionValue, Grid, minmax, View } from '@geti/ui';
import { ChevronLeft, ChevronRight } from '@geti/ui/icons';

import { CaptureFrameButton } from '../capture-frame-button.component';
import { Video } from '../video.component';
import { useActivateFrame } from './api/use-activate-frame.hook';
import { useGetFrames } from './api/use-frames.hook';
import { useGetActiveFrame } from './api/use-get-active-frame.hook';
import { FramesList } from './frames-list/frames-list.component';

import styles from './images-folder-stream.module.scss';

const useActiveFrameSelection = ({
    sourceId,
    activeFrameIdx,
    frames,
}: {
    sourceId: string;
    activeFrameIdx: number;
    frames: FrameType[];
}) => {
    const activateFrameMutation = useActivateFrame();
    const framesRef = useRef<HTMLDivElement>(null);
    const framesCount = frames.length;

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
        activateFrameMutation.mutate({
            sourceId,
            index: frameIdx,
            onSuccess: () => {
                scrollFrameIntoView(frameIdx);
            },
        });
    };

    const nextFrame = () => {
        const nextFrameIdx = activeFrameIdx + 1;

        if (nextFrameIdx >= framesCount) {
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
        activateFrame,
        nextFrame,
        prevFrame,
    };
};

interface ImagesFolderStreamProps {
    sourceId: string;
}

export const ImagesFolderStream = ({ sourceId }: ImagesFolderStreamProps) => {
    // const [promptMode] = usePromptMode();
    const { data: activeFrame } = useGetActiveFrame(sourceId);
    const activeFrameIdx = activeFrame.index;
    const { frames, fetchNextPage, fetchPreviousPage, framesCount } = useGetFrames(sourceId, activeFrameIdx);
    const { activateFrame, nextFrame, prevFrame, framesRef } = useActiveFrameSelection({
        sourceId,
        frames,
        activeFrameIdx,
    });

    const isPrevFrameButtonDisabled = activeFrameIdx === 0;
    const isNextFrameButtonDisabled = activeFrameIdx === framesCount - 1;

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
                paddingTop: dimensionValue('size-600'),
            }}
        >
            <ActionButton
                gridArea={'prev-btn'}
                alignSelf={'center'}
                UNSAFE_className={styles.button}
                isDisabled={isPrevFrameButtonDisabled}
                onPress={prevFrame}
                aria-label={'Previous Frame'}
            >
                <ChevronLeft />
            </ActionButton>
            <ActionButton
                gridArea={'next-btn'}
                alignSelf={'center'}
                UNSAFE_className={styles.button}
                isDisabled={isNextFrameButtonDisabled}
                onPress={nextFrame}
                aria-label={'Next Frame'}
            >
                <ChevronRight />
            </ActionButton>
            <View gridArea={'stream'}>
                <Video />
            </View>
            {/* TODO: Uncomment when we support text prompt
            {promptMode === 'visual' && (
                <View gridArea={'capture'} justifySelf={'center'}>
                    <CaptureFrameButton />
                </View>
            )}*/}
            <View gridArea={'capture'} justifySelf={'center'}>
                <CaptureFrameButton />
            </View>
            <View gridArea={'frames'}>
                <FramesList
                    ref={framesRef}
                    activeFrameIndex={activeFrameIdx}
                    onSetActiveFrame={activateFrame}
                    frames={frames}
                    fetchNextPage={fetchNextPage}
                    fetchPreviousPage={fetchPreviousPage}
                />
            </View>
        </Grid>
    );
};
