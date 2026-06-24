/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useProjectIdentifier } from '@/hooks';
import { getQueryKey } from '@/query-client';
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
import { useQueryClient } from '@tanstack/react-query';
import { useSpinDelay } from 'spin-delay';

import { useModelLoading, useModelStatus } from '../api/use-model-loading.hook';
import { useReloadProjectPipeline } from '../api/use-reload-project-pipleline.hook';

import classes from './model-loading-dialog.module.scss';

/**
 * Returns whether the blocking dialog is currently visible.
 *
 * Wraps the raw `loading` flag from the backend with `useSpinDelay` so that
 * very short loads don't trigger a UI flicker, and once shown the dialog
 * persists for a minimum duration.
 */
export const useShowModelLoadingDialog = (): boolean => {
    const loading = useModelLoading();
    return useSpinDelay(loading, { delay: 300, minDuration: 500 });
};

const ModelLoadingError = () => {
    const { data } = useModelStatus();
    const isError = data?.status === 'error';
    const reloadProjectPipelineMutation = useReloadProjectPipeline();
    const { projectId } = useProjectIdentifier();
    const queryClient = useQueryClient();

    const handleRetry = () => {
        reloadProjectPipelineMutation.mutate(
            {
                params: { path: { project_id: projectId } },
            },
            {
                onSuccess: () => {
                    queryClient.invalidateQueries({
                        queryKey: getQueryKey([
                            'get',
                            '/api/v1/projects/{project_id}/model-status',
                            { params: { path: { project_id: projectId } } },
                        ]),
                    });
                },
            }
        );
    };

    return (
        <DialogContainer onDismiss={() => {}} isDismissable={false} isKeyboardDismissDisabled>
            {isError && (
                <Dialog aria-label={'Model loading error'}>
                    <Heading level={3}>Model loading error</Heading>
                    <Divider />
                    <Content>
                        <Text UNSAFE_className={classes.errorMessage}>{data?.error_message}</Text>
                    </Content>
                    <ButtonGroup>
                        <Button onPress={handleRetry}>Retry</Button>
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
