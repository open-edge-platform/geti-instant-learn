/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content, Dialog, DialogContainer, Divider, Grid, Heading, minmax } from '@geti/ui';

import { useFullScreenMode } from '../../../annotator/actions/full-screen-mode.component';
import { CapturedFrameContent } from './captured-frame-content.component';

export const CapturedFrameFullScreen = () => {
    const { isFullScreenMode, setIsFullScreenMode } = useFullScreenMode();

    return (
        <DialogContainer type={'fullscreenTakeover'} onDismiss={() => setIsFullScreenMode(false)}>
            {isFullScreenMode && (
                <Dialog>
                    <Heading>Annotate frame</Heading>
                    <Divider />
                    <Content>
                        <Grid
                            height={'100%'}
                            width={'100%'}
                            areas={['labels', 'image', 'actions']}
                            rows={[minmax('size-500', 'auto'), minmax(0, '1fr'), 'size-500']}
                            UNSAFE_style={{
                                backgroundColor: 'var(--spectrum-global-color-gray-200)',
                            }}
                        >
                            <CapturedFrameContent />
                        </Grid>
                    </Content>
                </Dialog>
            )}
        </DialogContainer>
    );
};
