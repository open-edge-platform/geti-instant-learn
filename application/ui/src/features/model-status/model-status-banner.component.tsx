/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, InlineAlert, ProgressCircle, Text } from '@geti/ui';

import { useModelStatus } from './model-status-provider.component';

/**
 * Compact banner shown above the prompt panel while the model is being
 * (re)loaded or has failed to load. Renders nothing when the model is
 * READY or IDLE so it doesn't take up space during normal operation.
 */
export const ModelStatusBanner = () => {
    const { status, isBusy, isError } = useModelStatus();

    if (status === undefined) {
        return null;
    }

    if (isBusy) {
        return (
            <div role={'status'} aria-live={'polite'} aria-label={'Model loading'}>
                <Flex alignItems={'center'} gap={'size-150'}>
                    <ProgressCircle size={'S'} aria-label={'Loading'} isIndeterminate />
                    <Text>{status.message}</Text>
                </Flex>
            </div>
        );
    }

    if (isError) {
        return (
            <InlineAlert variant={'negative'} aria-label={'Model error'}>
                <Text>{status.message}</Text>
            </InlineAlert>
        );
    }

    return null;
};
