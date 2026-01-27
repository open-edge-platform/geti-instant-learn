/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { MQTTSinkType } from '@/api';
import { MQTT } from '@/icons';

import { PipelineEntityCard } from '../../pipeline-entity-card/pipeline-entity-card.component';

interface MQTTSinkCardProps {
    sink: MQTTSinkType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const MQTTSinkCard = ({ sink, menuItems, onAction }: MQTTSinkCardProps) => {
    const isActive = sink.active;

    const parameters = [
        `Name: ${sink.config.name}`,
        `Broker host: ${sink.config.broker_host}`,
        `Topic: ${sink.config.topic}`,
        `Broker port: ${sink.config.broker_port}`,
        `Auth required: ${sink.config.auth_required ? 'Yes' : 'No'}`,
    ];

    return (
        <PipelineEntityCard
            isActive={isActive}
            icon={<MQTT />}
            title={'MQTT'}
            menu={<PipelineEntityCard.Menu isActive={isActive} items={menuItems} onAction={onAction} />}
        >
            <PipelineEntityCard.Parameters parameters={parameters} />
        </PipelineEntityCard>
    );
};
