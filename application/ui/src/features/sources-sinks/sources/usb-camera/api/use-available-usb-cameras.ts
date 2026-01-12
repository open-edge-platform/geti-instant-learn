/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { USBCameraConfig } from '@geti-prompt/api';

import { useAvailableSources } from '../../api/use-available-sources';

export const useAvailableUsbCameras = () => {
    const { data } = useAvailableSources('usb_camera');
    const usbCameras = data as USBCameraConfig[];

    return { data: usbCameras };
};
