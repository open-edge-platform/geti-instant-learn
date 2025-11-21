/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { WebcamSourceType } from '@geti-prompt/api';
import { WebCam } from '@geti-prompt/icons';

import { SourceReadOnly } from '../source-read-only/source-read-only.component';

interface WebcamSourceReadOnlyProps {
    isActive: boolean;
    source: WebcamSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const WebcamSourceReadOnly = ({ isActive, source, onAction, menuItems }: WebcamSourceReadOnlyProps) => {
    const parameters = [`Device ID: ${source.config.device_id}`];

    return (
        <SourceReadOnly
            isActive={isActive}
            parameters={parameters}
            icon={<WebCam />}
            title={'Webcam'}
            menu={<SourceReadOnly.Menu isActive={isActive} items={menuItems} onAction={onAction} />}
        />
    );
};
