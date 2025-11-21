/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { WebcamSourceType } from '@geti-prompt/api';
import { WebCam } from '@geti-prompt/icons';

import { SourceCard } from '../source-card/source-card.component';

interface WebcamSourceCardProps {
    source: WebcamSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const WebcamSourceCard = ({ source, onAction, menuItems }: WebcamSourceCardProps) => {
    const parameters = [`Device ID: ${source.config.device_id}`];
    const isActiveSource = source.connected;

    return (
        <SourceCard
            isActive={isActiveSource}
            parameters={parameters}
            icon={<WebCam />}
            title={'Webcam'}
            menu={<SourceCard.Menu isActive={isActiveSource} items={menuItems} onAction={onAction} />}
        />
    );
};
