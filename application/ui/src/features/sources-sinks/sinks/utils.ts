/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { MQTTSinkType, SinkType } from '@geti-prompt/api';

export const isMQTTSink = (sink: SinkType): sink is MQTTSinkType => {
    return sink.config.sink_type === 'mqtt';
};

export type SinkViews = 'add' | 'edit' | 'list' | 'existing';
