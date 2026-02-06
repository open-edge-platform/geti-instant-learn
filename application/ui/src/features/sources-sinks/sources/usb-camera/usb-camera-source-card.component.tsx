/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { USBCameraSourceType } from '@/api';
import { UsbCamera } from '@/icons';

import { PipelineEntityCard } from '../../pipeline-entity-card/pipeline-entity-card.component';

interface UsbCameraSourceCardProps {
    source: USBCameraSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const UsbCameraSourceCard = ({ source, onAction, menuItems }: UsbCameraSourceCardProps) => {
    const parameters = [`Device: ${source.config.name}`];
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
