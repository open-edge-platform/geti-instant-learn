/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, ProgressCircle, StatusLight, Tooltip, TooltipTrigger } from '@geti/ui';

import { useModelStatus } from './model-status-provider.component';

type Variant = 'info' | 'positive' | 'neutral' | 'negative';

const SHORT_LABELS: Record<string, string> = {
    idle: 'Idle',
    loading_reference_batch: 'Building prompts…',
    loading_model: 'Loading…',
    ready: 'Ready',
    error: 'Error',
};

const VARIANTS: Record<string, Variant> = {
    idle: 'neutral',
    loading_reference_batch: 'info',
    loading_model: 'info',
    ready: 'positive',
    error: 'negative',
};

/**
 * Compact, always-visible model status indicator meant to sit next to the
 * model picker. Shows a short label + colored ``StatusLight`` for at-a-glance
 * state, with the full backend message available on hover via tooltip.
 *
 * Renders nothing only while the very first snapshot is still loading; once
 * any state has been received it stays visible (including ``READY``) so the
 * user always sees the current model state.
 */
export const ModelStatusBanner = () => {
    const { status, isBusy } = useModelStatus();

    if (status === undefined) {
        return null;
    }

    const shortLabel = SHORT_LABELS[status.state] ?? status.state;
    const variant = VARIANTS[status.state] ?? 'neutral';

    return (
        <TooltipTrigger>
            <div role={'status'} aria-label={`Model ${shortLabel}`} aria-live={'polite'}>
                <Flex alignItems={'center'} gap={'size-100'}>
                    {isBusy && <ProgressCircle size={'S'} aria-label={'Loading'} isIndeterminate />}
                    <StatusLight variant={variant}>{shortLabel}</StatusLight>
                </Flex>
            </div>
            <Tooltip>{status.message}</Tooltip>
        </TooltipTrigger>
    );
};
