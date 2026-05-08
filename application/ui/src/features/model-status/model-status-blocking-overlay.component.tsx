/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelState, ModelStatusType } from '@/api';
import { Flex, Heading, ProgressCircle, Text, View } from '@geti/ui';
import { FocusScope } from '@react-aria/focus';

import { useModelStatus } from './model-status-provider.component';
import { useDebouncedVisibility } from './use-debounced-visibility.hook';

interface BlockingCopy {
    title: string;
    hint: string;
}

const BLOCKING_COPY: Partial<Record<ModelState, BlockingCopy>> = {
    loading_model: {
        title: 'Loading model…',
        hint: 'Please wait — this may take a moment on first run while weights are downloaded.',
    },
    loading_reference_batch: {
        title: 'Building reference batch from your prompts…',
        hint: 'The model is preparing to run inference. This should only take a few seconds.',
    },
};

const buildTarget = (status: ModelStatusType | undefined): string => {
    if (!status) return '';
    const parts = [status.model_name, status.device && `on ${status.device}`].filter(Boolean);
    return parts.join(' ');
};

export const ModelStatusBlockingOverlay = () => {
    const { status, isBusy } = useModelStatus();
    const visible = useDebouncedVisibility(isBusy, 200, 400);

    if (!visible) return null;

    const state = status?.state;
    const copy = (state && BLOCKING_COPY[state]) ?? BLOCKING_COPY.loading_model!;
    const target = buildTarget(status);

    return (
        <FocusScope contain restoreFocus>
            <div
                role={'alertdialog'}
                aria-modal={'true'}
                aria-live={'assertive'}
                aria-label={copy.title}
                style={{
                    position: 'fixed',
                    inset: 0,
                    zIndex: 1000,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: 'rgba(0, 0, 0, 0.45)',
                    pointerEvents: 'auto',
                }}
            >
                <View
                    backgroundColor={'gray-50'}
                    padding={'size-600'}
                    borderRadius={'large'}
                    maxWidth={'size-6000'}
                    UNSAFE_style={{ textAlign: 'center' }}
                >
                    <Flex direction={'column'} alignItems={'center'} gap={'size-300'}>
                        <ProgressCircle size={'L'} aria-label={'Loading'} isIndeterminate />
                        <Heading level={3}>{copy.title}</Heading>
                        {target && (
                            <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>{target}</Text>
                        )}
                        {status?.message && (
                            <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-600)', fontSize: '14px' }}>
                                {status.message}
                            </Text>
                        )}
                        <Text
                            UNSAFE_style={{
                                color: 'var(--spectrum-global-color-gray-600)',
                                fontSize: '12px',
                                fontStyle: 'italic',
                            }}
                        >
                            {copy.hint}
                        </Text>
                    </Flex>
                </View>
            </div>
        </FocusScope>
    );
};
