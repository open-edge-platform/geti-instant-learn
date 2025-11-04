/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Divider, Flex } from '@geti/ui';
import { SegmentAnythingIcon } from '@geti/ui/icons';
import { useHotkeys } from 'react-hotkeys-hook';

import { HOTKEYS } from '../hotkeys/hotkeys';
import { useAnnotator } from '../providers/annotator-provider.component';
import { ToolConfig } from './interface';
import { Tools } from './tools.component';

const TOOLS: ToolConfig[] = [{ type: 'sam', icon: SegmentAnythingIcon }];

export const AnnotatorTools = () => {
    const { activeTool, setActiveTool } = useAnnotator();

    useHotkeys(HOTKEYS.enableSam, () => setActiveTool('sam'), [setActiveTool]);

    return (
        <Flex alignItems={'center'} marginEnd={'auto'}>
            <Tools tools={TOOLS} activeTool={activeTool} setActiveTool={setActiveTool} />
            <Divider size='S' />
        </Flex>
    );
};
