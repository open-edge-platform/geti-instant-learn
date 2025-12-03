/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { isInteger } from 'lodash-es';

export const isDeviceIdValid = (deviceId: string) => {
    return isInteger(Number(deviceId));
};

export const validateDeviceId = (deviceId: string) =>
    isDeviceIdValid(deviceId) ? true : 'Device ID must be an integer';
