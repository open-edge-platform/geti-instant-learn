/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, useState } from 'react';

import { useCurrentProject } from '@geti-prompt/hooks';
import { Wand } from '@geti-prompt/icons';
import { Flex, ToggleButton, View } from '@geti/ui';
import { clsx } from 'clsx';
import { Panel, Separator, usePanelRef } from 'react-resizable-panels';

import { Prompt } from '../../features/prompts/prompt.component';

import styles from './sidebar.module.scss';

type TabItem = { label: string; icon: ReactNode; content: ReactNode };

interface TabProps {
    tabs: TabItem[];
    selectedTab: string | null;
}

const SidebarTabs = ({ tabs, selectedTab }: TabProps) => {
    const [tab, setTab] = useState<string | null>(selectedTab);
    const [isAnimating, setIsAnimating] = useState(false);
    const ref = usePanelRef();

    const activeTab = tabs.find(({ label }) => label === tab);
    const content = activeTab?.content;

    const [displayContent, setDisplayContent] = useState<boolean>(activeTab !== undefined);

    /*const handleTabChange = (newTab: string): void => {
        setIsAnimating(true);

        if (newTab === tab) {
            setTab(null);
            ref.current?.collapse();
        } else {
            setTab(newTab);
            ref.current?.isCollapsed() && ref.current?.expand();
        }

        setTimeout(() => {
            setIsAnimating(false);
        }, 300);
    };*/

    const handleTabChange = (newTab: string): void => {
        setIsAnimating(true);

        if (newTab === tab) {
            setTab(null);

            setTimeout(() => {
                setDisplayContent(false);
            }, 250);
        } else {
            setTab(newTab);
            setDisplayContent(true);
        }

        setTimeout(() => {
            setIsAnimating(false);
        }, 250);
    };

    return (
        <>
            {displayContent && (
                <>
                    <Separator className={clsx(styles.separator, styles.separatorEnabled)} />
                    <Panel
                        data-isanimating={isAnimating}
                        data-collapsed={content === undefined}
                        id={'sidebar'}
                        defaultSize={'35%'}
                        minSize={'30%'}
                        className={clsx(styles.sidebarContent, {
                            [styles.sidebarContentGoingToCollapse]: content === undefined,
                        })}
                    >
                        {content}
                    </Panel>
                </>
            )}
            {/*<Separator
                className={clsx(styles.separator, {
                    [styles.separatorEnabled]: content !== undefined,
                    [styles.separatorDisabled]: content === undefined,
                })}
            />
            <Panel
                data-isanimating={isAnimating}
                id={'sidebar'}
                defaultSize={'35%'}
                minSize={'30%'}
                collapsible
                panelRef={ref}
                className={styles.sidebarContent}
            >
                {content}
            </Panel>*/}
            <View backgroundColor={'gray-200'} padding={'size-100'}>
                <Flex direction={'column'} height={'100%'} alignItems={'center'} gap={'size-100'}>
                    {tabs.map(({ label, icon }) => (
                        <ToggleButton
                            key={label}
                            isQuiet
                            isSelected={label === tab}
                            onChange={() => handleTabChange(label)}
                            UNSAFE_className={styles.toggleButton}
                            aria-label={`Toggle ${label} tab`}
                        >
                            {icon}
                        </ToggleButton>
                    ))}
                </Flex>
            </View>
        </>
    );
};

export const Sidebar = () => {
    const { data } = useCurrentProject();

    const TABS: TabItem[] = [
        { label: 'Prompt', icon: <Wand />, content: <Prompt /> },
        //{ label: 'Model statistics', icon: <GraphChart />, content: <Graphs /> },
    ];

    return (
        // When the project activity status changes (e.g. from active to inactive, we want to toggle and disable
        // sidebar).
        <SidebarTabs key={`${data.id}-${data.active}`} tabs={TABS} selectedTab={TABS[0].label} />
    );
};
