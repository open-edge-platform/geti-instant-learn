/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content, Dialog, DialogContainer, Flex, Heading, ProgressCircle, Text } from '@geti/ui';
import { useSpinDelay } from 'spin-delay';

import { useModelLoading } from './use-model-loading.hook';

/**
 * Returns whether the blocking dialog is currently visible.
 *
 * Wraps the raw `loading` flag from the backend with `useSpinDelay` so that
 * very short loads don't trigger a UI flicker, and once shown the dialog
 * persists for a minimum duration.
 */
export const useShowModelLoadingDialog = (): boolean => {
    const loading = useModelLoading();
    return useSpinDelay(loading, { delay: 300, minDuration: 400 });
};

/**
 * Non-dismissable blocking dialog shown while the inference model is being
 * (re)prepared. The user cannot interact with the rest of the UI until the
 * model is ready.
 */
export const ModelLoadingDialog = () => {
    const visible = useShowModelLoadingDialog();

    return (
        <DialogContainer
            onDismiss={() => {
                /* no-op — dialog is intentionally non-dismissable */
            }}
            isDismissable={false}
            isKeyboardDismissDisabled
        >
            {visible && (
                <Dialog aria-label={'Loading model'} size={'S'}>
                    <Content>
                        <Flex direction={'column'} alignItems={'center'} gap={'size-300'}>
                            <ProgressCircle
                                size={'L'}
                                aria-label={'Loading'}
                                isIndeterminate
                                UNSAFE_style={{ flexShrink: 0 }}
                            />
                            <Heading level={3} UNSAFE_style={{ textAlign: 'center' }}>
                                Loading model…
                            </Heading>
                            <Text
                                UNSAFE_style={{
                                    color: 'var(--spectrum-global-color-gray-700)',
                                    fontStyle: 'italic',
                                    textAlign: 'center',
                                }}
                            >
                                Please wait — this may take a moment on first run while weights are downloaded.
                            </Text>
                        </Flex>
                    </Content>
                </Dialog>
            )}
        </DialogContainer>
    );
};
