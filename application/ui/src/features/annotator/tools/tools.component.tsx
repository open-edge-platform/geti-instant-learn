/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton } from '@geti/ui';
import { Fragment } from 'react/jsx-runtime';

import type { ToolConfig, ToolType } from './interface';

interface ToolsProps {
    tools: ToolConfig[];
    activeTool: ToolType | null;
    setActiveTool: (tool: ToolType) => void;
}
export const Tools = ({ tools, activeTool, setActiveTool }: ToolsProps) => {
    if (tools.length === 0) {
        return null;
    }

    return (
        <>
            {tools.map((tool) => (
                <Fragment key={tool.type}>
                    <ActionButton
                        aria-label={`Select ${tool.type} Tool`}
                        isQuiet={activeTool !== tool.type}
                        aria-pressed={activeTool === tool.type}
                        onPress={() => setActiveTool(tool.type)}
                    >
                        <tool.icon data-tool={tool.type} />
                    </ActionButton>
                </Fragment>
            ))}
        </>
    );
};
