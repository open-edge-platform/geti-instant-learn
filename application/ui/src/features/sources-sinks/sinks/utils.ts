/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { MQTTSinkType, SinkConfig } from '@geti-prompt/api';

export const isMQTTSink = (sink: SinkConfig): sink is MQTTSinkType => {
    return sink.config.sink_type === 'mqtt';
};

export type SinkViews = 'add' | 'edit' | 'list' | 'existing';
