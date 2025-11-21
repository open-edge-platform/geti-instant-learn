/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { WebcamSourceType } from '@geti-prompt/api';
import { WebCam } from '@geti-prompt/icons';

import { SourceCard } from '../source-card/source-card.component';

interface WebcamSourceCardProps {
    isActive: boolean;
    source: WebcamSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const WebcamSourceCard = ({ isActive, source, onAction, menuItems }: WebcamSourceCardProps) => {
    const parameters = [`Device ID: ${source.config.device_id}`];

    return (
        <SourceCard
            isActive={isActive}
            parameters={parameters}
            icon={<WebCam />}
            title={'Webcam'}
            menu={<SourceCard.Menu isActive={isActive} items={menuItems} onAction={onAction} />}
        />
    );
};
