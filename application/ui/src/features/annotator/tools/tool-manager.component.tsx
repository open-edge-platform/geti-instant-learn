/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useAnnotator } from '../providers/annotator-provider.component';
import { SegmentAnythingTool } from './segment-anything-tool/segment-anything-tool.component';

export const ToolManager = () => {
    const { activeTool } = useAnnotator();

    if (activeTool === 'sam') {
        return <SegmentAnythingTool />;
    }

    return null;
};
