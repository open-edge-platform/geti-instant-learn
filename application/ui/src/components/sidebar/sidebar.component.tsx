/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, useState } from 'react';

import { Wand } from '@geti-prompt/icons';
import { Flex, Grid, ToggleButton, View } from '@geti/ui';

import { useCurrentProject } from '../../features/project/hooks/use-current-project.hook';
import { Prompt } from '../../features/prompts/prompt.component';

import styles from './sidebar.module.scss';

type TabItem = { label: string; icon: ReactNode; content: ReactNode; isDisabled?: boolean };

interface TabProps {
    tabs: TabItem[];
    selectedTab: string | null;
}

const SidebarTabs = ({ tabs, selectedTab }: TabProps) => {
    const [tab, setTab] = useState<string | null>(selectedTab);

    // If no tab is selected but we have enabled tabs, select the first one
    const activeTab = tab ?? tabs.find((t) => !t.isDisabled)?.label ?? null;

    const content = tabs.find(({ label }) => label === activeTab)?.content;

    return (
        <Grid UNSAFE_className={styles.container} columns={['1fr', 'size-600']} height={'100%'} data-expanded='true'>
            <View gridColumn={'1/2'} UNSAFE_className={styles.sidebarContent}>
                {content}
            </View>
            <View gridColumn={'2/3'} backgroundColor={'gray-200'} padding={'size-100'}>
                <Flex direction={'column'} height={'100%'} alignItems={'center'} gap={'size-100'}>
                    {tabs.map(({ label, icon, isDisabled }) => (
                        <ToggleButton
                            key={label}
                            isQuiet
                            isSelected={label === activeTab}
                            onChange={() => setTab(label)}
                            UNSAFE_className={styles.toggleButton}
                            aria-label={`${label} tab`}
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
    const { data } = useCurrentProject();

    const TABS: TabItem[] = [{ label: 'Prompt', icon: <Wand />, content: <Prompt />, isDisabled: !data.active }];

    return (
        // When the project activity status changes (e.g. from active to inactive, we want to toggle and disable
        // sidebar).
        <SidebarTabs key={`${data.id}-${data.active}`} tabs={TABS} selectedTab={data.active ? TABS[0].label : null} />
    );
};
