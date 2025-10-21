/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
export interface RegionOfInterest {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface ImageItem {
    url: string;
    size: {
        x: number;
        y: number;
    };
}
