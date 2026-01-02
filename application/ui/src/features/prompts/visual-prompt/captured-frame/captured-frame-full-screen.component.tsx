/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import {
    ActionButton,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Flex,
    Grid,
    Heading,
    minmax,
    View,
} from '@geti/ui';
import { Close } from '@geti/ui/icons';

import { useFullScreenMode } from '../../../annotator/actions/full-screen-mode.component';
import { SavePrompt } from '../save-prompt/save-prompt.component';
import { CapturedFrameContent } from './captured-frame-content.component';

export const CapturedFrameFullScreen = () => {
    const { isFullScreenMode, setIsFullScreenMode } = useFullScreenMode();

    const closeFullScreenMode = () => setIsFullScreenMode(false);

    return (
        <DialogContainer type={'fullscreenTakeover'} onDismiss={closeFullScreenMode}>
            {isFullScreenMode && (
                <Dialog>
                    <Heading>Prompt builder</Heading>
                    <Divider />
                    <Content>
                        <Flex direction={'column'} gap={'size-200'} height={'100%'}>
                            <Grid
                                minHeight={0}
                                flex={1}
                                width={'100%'}
                                areas={['labels', 'image', 'actions']}
                                rows={[minmax('size-500', 'auto'), minmax(0, '1fr'), 'size-500']}
                                UNSAFE_style={{
                                    backgroundColor: 'var(--spectrum-global-color-gray-200)',
                                }}
                            >
                                <CapturedFrameContent />
                            </Grid>
                            <View marginStart={'auto'}>
                                <SavePrompt />
                            </View>
                        </Flex>
                    </Content>
                    <ButtonGroup>
                        <ActionButton isQuiet aria-label={'Close full screen'} onPress={closeFullScreenMode}>
                            <Close />
                        </ActionButton>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogContainer>
    );
};
