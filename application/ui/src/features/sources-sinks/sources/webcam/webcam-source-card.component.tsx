/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { WebcamSourceType } from '@geti-prompt/api';
import { WebCam } from '@geti-prompt/icons';

import { PipelineEntityCard } from '../../pipeline-entity-card/pipeline-entity-card.component';

interface WebcamSourceCardProps {
    source: WebcamSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const WebcamSourceCard = ({ source, onAction, menuItems }: WebcamSourceCardProps) => {
    const parameters = [`Device ID: ${source.config.device_id}`];
    const isActiveSource = source.active;

    return (
        <PipelineEntityCard
            isActive={isActiveSource}
            icon={<WebCam />}
            title={'Webcam'}
            menu={<PipelineEntityCard.Menu isActive={isActiveSource} items={menuItems} onAction={onAction} />}
        >
            <PipelineEntityCard.Parameters parameters={parameters} />
        </PipelineEntityCard>
    );
};
