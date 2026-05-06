/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, ProgressCircle, StatusLight, Tooltip, TooltipTrigger } from '@geti/ui';

import { ModelState, ModelStatusType } from '@/api';

import { useModelStatus } from './model-status-provider.component';

type Variant = 'info' | 'positive' | 'neutral' | 'negative';

const VARIANTS: Record<ModelState, Variant> = {
    idle: 'neutral',
    loading_reference_batch: 'info',
    loading_model: 'info',
    ready: 'positive',
    error: 'negative',
};

/**
 * Build the visible status label. Always tries to surface the model name and
 * resolved device when known so the user can see *which* model is active and
 * *where* it runs without hovering for the tooltip.
 */
const buildLabel = (status: ModelStatusType): string => {
    const { state, model_name: modelName, device } = status;
    const target = [modelName, device && `on ${device}`].filter(Boolean).join(' ');

    switch (state) {
        case 'ready':
            return target ? `Ready · ${target}` : 'Ready';
        case 'loading_model':
            return target ? `Loading ${target}…` : 'Loading…';
        case 'loading_reference_batch':
            return target ? `Building prompts · ${target}` : 'Building prompts…';
        case 'error':
            return target ? `Error · ${target}` : 'Error';
        case 'idle':
        default:
            return 'Idle';
    }
};

/**
 * Compact, always-visible model status indicator meant to sit next to the
 * model picker. Shows the current state plus model name and device when
 * available, with the full backend message on hover via tooltip.
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

    const label = buildLabel(status);
    const variant = VARIANTS[status.state] ?? 'neutral';

    return (
        <TooltipTrigger>
            <div role={'status'} aria-label={`Model ${label}`} aria-live={'polite'}>
                <Flex alignItems={'center'} gap={'size-100'}>
                    {isBusy && <ProgressCircle size={'S'} aria-label={'Loading'} isIndeterminate />}
                    <StatusLight variant={variant}>{label}</StatusLight>
                </Flex>
            </div>
            <Tooltip>{status.message}</Tooltip>
        </TooltipTrigger>
    );
};
