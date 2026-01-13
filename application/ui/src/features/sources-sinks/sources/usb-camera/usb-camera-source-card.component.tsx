/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { USBCameraSourceType } from '@geti-prompt/api';
import { UsbCamera } from '@geti-prompt/icons';

import { PipelineEntityCard } from '../../pipeline-entity-card/pipeline-entity-card.component';

interface UsbCameraSourceCardProps {
    source: USBCameraSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const UsbCameraSourceCard = ({ source, onAction, menuItems }: UsbCameraSourceCardProps) => {
    const parameters = [`Device ID: ${source.config.device_id}`];
    const isActiveSource = source.active;

    return (
        <PipelineEntityCard
            isActive={isActiveSource}
            icon={<UsbCamera />}
            title={'USB Camera'}
            menu={<PipelineEntityCard.Menu isActive={isActiveSource} items={menuItems} onAction={onAction} />}
        >
            <PipelineEntityCard.Parameters parameters={parameters} />
        </PipelineEntityCard>
    );
};
