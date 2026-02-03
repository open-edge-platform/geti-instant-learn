/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { VideoFileSourceType } from '@/api';
import { VideoFile } from '@/icons';

import { PipelineEntityCard } from '../../pipeline-entity-card/pipeline-entity-card.component';

interface VideoFileCardProps {
    source: VideoFileSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const VideoFileCard = ({ source, onAction, menuItems }: VideoFileCardProps) => {
    const isActiveSource = source.active;
    const parameters = [`File path: ${source.config.video_path}`];

    return (
        <PipelineEntityCard
            isActive={isActiveSource}
            icon={<VideoFile />}
            title={'Video file'}
            menu={<PipelineEntityCard.Menu isActive={isActiveSource} items={menuItems} onAction={onAction} />}
        >
            <PipelineEntityCard.Parameters parameters={parameters} />
        </PipelineEntityCard>
    );
};
