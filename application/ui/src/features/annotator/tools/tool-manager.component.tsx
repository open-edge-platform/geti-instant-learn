/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useAnnotator } from '../providers/annotator-provider.component';

export const ToolManager = () => {
    const { activeTool } = useAnnotator();

    if (activeTool === 'sam') {
        // TODO: Import the actual tool
        // return <SegmentAnythingTool />;
    }

    return null;
};
