/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { USBCameraConfig } from '@geti-prompt/api';

import { useAvailableSources } from '../../api/use-available-sources';
import { isUsbCameraConfig } from '../../utils';

export const useAvailableUsbCameras = (): { data: USBCameraConfig[] } => {
    const { data } = useAvailableSources('usb_camera');

    const cameraDevices = data.filter(isUsbCameraConfig);

    return { data: cameraDevices };
};
