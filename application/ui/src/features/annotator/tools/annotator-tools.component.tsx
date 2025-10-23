/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Divider } from '@geti/ui';
import { BoundingBox, SegmentAnythingIcon, Selector } from '@geti/ui/icons';

import { useAnnotator } from '../providers/annotator-provider.component';
import { ToolConfig } from './interface';
import { Tools } from './tools.component';

const TOOLS: ToolConfig[] = [
    { type: 'selection', icon: Selector },
    { type: 'bounding-box', icon: BoundingBox },
    { type: 'sam', icon: SegmentAnythingIcon },
];

export const AnnotatorTools = () => {
    const { activeTool, setActiveTool } = useAnnotator();

    return (
        <>
            <Tools tools={TOOLS} activeTool={activeTool} setActiveTool={setActiveTool} />
            <Divider size='S' />
        </>
    );
};
