/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CapturedFrameContent } from './captured-frame-content.component';
import { CapturedFrameFullScreen } from './captured-frame-full-screen.component';

export const CapturedFrame = ({ frameId }: { frameId: string }) => {
    return (
        <>
            <CapturedFrameContent frameId={frameId} />
            <CapturedFrameFullScreen frameId={frameId} />
        </>
    );
};
