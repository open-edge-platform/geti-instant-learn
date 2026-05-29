/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { MQTTSinkType } from '@/api';

export const getMockedMQTTSink = (overrides: Partial<MQTTSinkType> = {}): MQTTSinkType => {
    return {
        id: overrides.id ?? 'mqtt-sink-id',
        active: overrides.active ?? true,
        config: {
            sink_type: 'mqtt',
            name: overrides.config?.name ?? 'My MQTT Sink',
            broker_host: overrides.config?.broker_host ?? 'mqtt.example.com',
            topic: overrides.config?.topic ?? 'test/topic',
            broker_port: overrides.config?.broker_port ?? 1883,
            auth_required: overrides.config?.auth_required ?? true,
        },
    };
};
