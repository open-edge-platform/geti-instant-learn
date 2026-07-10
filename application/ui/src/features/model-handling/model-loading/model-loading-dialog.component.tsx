/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Flex,
    Heading,
    ProgressCircle,
    Text,
} from '@geti/ui';
import { Link } from 'react-router-dom';
import { useSpinDelay } from 'spin-delay';

import { useModelLoading, useModelStatus } from '../api/use-model-loading.hook';

import classes from './model-loading-dialog.module.scss';

const SPIN_DELAY_MS = 300;
const SPIN_MIN_DURATION_MS = 500;

/**
 * Returns whether the blocking dialog is currently visible.
 *
 * Wraps the raw `loading` flag from the backend with `useSpinDelay` so that
 * very short loads don't trigger a UI flicker, and once shown the dialog
 * persists for a minimum duration.
 */
export const useShowModelLoadingDialog = (): boolean => {
    const loading = useModelLoading();
    return useSpinDelay(loading, { delay: SPIN_DELAY_MS, minDuration: SPIN_MIN_DURATION_MS });
};

const ModelLoadingError = () => {
    const { data } = useModelStatus();
    const isError = data?.status === 'error';
    const [isClosed, setIsClosed] = useState(false);

    const handleClose = () => setIsClosed(true);

    return (
        <DialogContainer onDismiss={handleClose} isDismissable={false} isKeyboardDismissDisabled>
            {isError && !isClosed && (
                <Dialog aria-label={'Model loading error'}>
                    <Heading level={3}>Model loading error</Heading>
                    <Divider />
                    <Content>
                        <Flex direction={'column'} gap={'size-50'}>
                            <Text UNSAFE_className={classes.errorMessage}>{data.error_message}</Text>
                            <Text UNSAFE_className={classes.errorMessage}>
                                See{' '}
                                <Link to={data.error_doc_url ?? ''} target='_blank'>
                                    documentation
                                </Link>
                                .
                            </Text>
                        </Flex>
                    </Content>
                    <ButtonGroup>
                        <Button onPress={handleClose}>Close</Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogContainer>
    );
};

/**
 * Non-dismissable blocking dialog shown while the inference model is being
 * (re)prepared. The user cannot interact with the rest of the UI until the
 * model is ready.
 */
export const ModelLoadingDialog = () => {
    const visible = useShowModelLoadingDialog();

    if (!visible) {
        return <ModelLoadingError />;
    }

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
                    <Heading level={3}>Loading model…</Heading>
                    <Divider />
                    <Content>
                        <Flex direction={'column'} alignItems={'center'} gap={'size-300'}>
                            <ProgressCircle
                                size={'L'}
                                aria-label={'Loading'}
                                isIndeterminate
                                UNSAFE_style={{ flexShrink: 0 }}
                            />

                            <Text
                                UNSAFE_style={{
                                    color: 'var(--spectrum-global-color-gray-700)',
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
