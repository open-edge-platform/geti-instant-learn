/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SinksListType } from '@/api';

import { getMockedMQTTSink } from '../../src/test-utils/mocks/mock-mqtt-sink';

export const ACTIVE_MQTT_SINK = getMockedMQTTSink({ active: true });
export const INACTIVE_MQTT_SINK = getMockedMQTTSink({ active: false });

export const mockSinksResponse = (sinks: SinksListType['sinks']): SinksListType => ({
    sinks,
    pagination: { count: sinks.length, total: sinks.length, offset: 0, limit: 10 },
});
