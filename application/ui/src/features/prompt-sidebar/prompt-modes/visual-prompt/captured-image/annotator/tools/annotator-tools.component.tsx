/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Divider } from '@geti/ui';

export const AnnotatorTools = () => {
    //const projectTask = useProjectTask();
    const { activeTool, setActiveTool } = useAnnotator();

    const availableTools = [{ type: 'bounding-box', icon: BoundingBox }];
    return (
        <>
            <Tools tools={availableTools} activeTool={activeTool} setActiveTool={setActiveTool} />
            {availableTools.length > 0 && <Divider size='S' />}
        </>
    );
};
