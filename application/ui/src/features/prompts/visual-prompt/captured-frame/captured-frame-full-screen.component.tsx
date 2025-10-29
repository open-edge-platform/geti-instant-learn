/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content, Dialog, DialogContainer, Divider, Grid, Heading, minmax } from '@geti/ui';

import { CapturedFrameContent } from './captured-frame.component';

interface CapturedFrameFullScreenProps {
    isFullScreenMode: boolean;
    onFullScreenModeChange: (isFullScreenMode: boolean) => void;
    frameId: string;
}

export const CapturedFrameFullScreen = ({
    isFullScreenMode,
    onFullScreenModeChange,
    frameId,
}: CapturedFrameFullScreenProps) => {
    return (
        <DialogContainer type={'fullscreenTakeover'} onDismiss={() => onFullScreenModeChange(false)}>
            {isFullScreenMode && (
                <Dialog>
                    <Heading>Annotate frame</Heading>
                    <Divider />
                    <Content>
                        <Grid
                            height={'100%'}
                            width={'100%'}
                            areas={['labels', 'image', 'actions']}
                            rows={['size-500', minmax(0, '1fr'), 'size-500']}
                            UNSAFE_style={{
                                backgroundColor: 'var(--spectrum-global-color-gray-200)',
                            }}
                        >
                            <CapturedFrameContent
                                frameId={frameId}
                                isFullScreenMode={isFullScreenMode}
                                onFullScreenModeChange={onFullScreenModeChange}
                            />
                        </Grid>
                    </Content>
                </Dialog>
            )}
        </DialogContainer>
    );
};
