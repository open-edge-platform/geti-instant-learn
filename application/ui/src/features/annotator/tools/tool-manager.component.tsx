/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ToolType } from './interface';
import { SegmentAnythingTool } from './segment-anything-tool/segment-anything-tool.component';

export const ToolManager = ({ activeTool }: { activeTool: ToolType | null }) => {
    if (activeTool === 'sam') {
        return <SegmentAnythingTool />;
    }

    return null;
};
