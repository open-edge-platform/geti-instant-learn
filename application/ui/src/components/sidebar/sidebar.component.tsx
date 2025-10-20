/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, useState } from 'react';

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Wand } from '@geti-prompt/icons';
import { Flex, Grid, ToggleButton, View } from '@geti/ui';

import { PromptSidebar } from '../../features/prompt-sidebar/prompt-sidebar.component';

import styles from './sidebar.module.scss';

type TabItem = { label: string; icon: ReactNode; content: ReactNode; isDisabled?: boolean };

interface TabProps {
    tabs: TabItem[];
    selectedTab: string | null;
}

const SidebarTabs = ({ tabs, selectedTab }: TabProps) => {
    const [tab, setTab] = useState<string | null>(selectedTab);

    const gridTemplateColumns = tab !== null ? ['clamp(size-4600, 35vw, 40rem)', 'size-600'] : ['0px', 'size-600'];

    const content = tabs.find(({ label }) => label === tab)?.content;

    return (
        <Grid
            gridArea={'sidebar'}
            UNSAFE_className={styles.container}
            columns={gridTemplateColumns}
            data-expanded={tab !== null}
        >
            <View gridColumn={'1/2'} UNSAFE_className={styles.sidebarContent}>
                {content}
            </View>
            <View gridColumn={'2/3'} backgroundColor={'gray-200'} padding={'size-100'}>
                <Flex direction={'column'} height={'100%'} alignItems={'center'} gap={'size-100'}>
                    {tabs.map(({ label, icon, isDisabled }) => (
                        <ToggleButton
                            key={label}
                            isQuiet
                            isSelected={label === tab}
                            onChange={() => setTab(label === tab ? null : label)}
                            UNSAFE_className={styles.toggleButton}
                            aria-label={`Toggle ${label} tab`}
                            isDisabled={isDisabled}
                        >
                            {icon}
                        </ToggleButton>
                    ))}
                </Flex>
            </View>
        </Grid>
    );
};

export const Sidebar = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });

    const TABS: TabItem[] = [{ label: 'Prompt', icon: <Wand />, content: <PromptSidebar />, isDisabled: !data.active }];

    return (
        // When the project activity status changes (e.g. from active to inactive, we want to toggle and disable
        // sidebar).
        <SidebarTabs key={`${projectId}-${data.active}`} tabs={TABS} selectedTab={data.active ? TABS[0].label : null} />
    );
};
