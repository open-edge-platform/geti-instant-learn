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
    title: (modelLabel: string) => string;
    hint: string;
}

const BLOCKING_COPY: Partial<Record<ModelState, BlockingCopy>> = {
    loading_model: {
        title: (model) => (model ? `Loading ${model}…` : 'Loading model…'),
        hint: 'Please wait — this may take a moment on first run while weights are downloaded.',
    },
    loading_reference_batch: {
        title: (model) => (model ? `Preparing ${model} for inference…` : 'Preparing the model for inference…'),
        hint: 'Building a reference batch from your prompts. This should only take a few seconds.',
    },
};

/**
 * Human-friendly display labels for known model types.
 * These match the identifiers sent by the backend in status messages.
 */
const MODEL_TYPE_LABELS: Record<string, string> = {
    matcher: 'Matcher',
    soft_matcher: 'Soft Matcher',
    perdino: 'PerDINO',
    sam3: 'SAM3',
};

/**
 * Format the raw model_name (e.g. "soft_matcher") into a human-friendly label
 * (e.g. "Soft Matcher"). Uses a known-labels map, falling back to title-casing.
 */
const formatModelName = (raw: string): string =>
    MODEL_TYPE_LABELS[raw] ??
    raw
        .split('_')
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');

/**
 * Build a short, human-friendly model label like "Soft Matcher (XPU)".
 */
const buildModelLabel = (status: ModelStatusType | undefined): string => {
    if (!status) return '';
    const parts: string[] = [];
    if (status.model_name) {
        parts.push(formatModelName(status.model_name));
    }
    if (status.device) {
        parts.push(`(${status.device.toUpperCase()})`);
    }
    return parts.join(' ');
};

export const ModelStatusBlockingOverlay = () => {
    const { status, isBusy } = useModelStatus();
    const visible = useDebouncedVisibility(isBusy, 200, 400);

    if (!visible) return null;

    const state = status?.state;
    const copy = (state && BLOCKING_COPY[state]) ?? BLOCKING_COPY.loading_model!;
    const modelLabel = buildModelLabel(status);
    const title = copy.title(modelLabel);

    return (
        <FocusScope contain restoreFocus>
            <div
                role={'alertdialog'}
                aria-modal={'true'}
                aria-live={'assertive'}
                aria-label={title}
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
                        <Heading level={3}>{title}</Heading>
                        <Text
                            UNSAFE_style={{
                                color: 'var(--spectrum-global-color-gray-600)',
                                fontSize: '13px',
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
