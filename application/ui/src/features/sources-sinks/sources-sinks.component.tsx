/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Suspense } from 'react';

import { InputOutput } from '@/icons';
import {
    Button,
    Content,
    Dialog,
    DialogTrigger,
    Flex,
    Item,
    Loading,
    TabList,
    TabPanels,
    Tabs,
    Text,
    View,
} from '@geti/ui';

import { usePrefetchPipelineConfiguration } from './api/use-prefetch-pipeline-configuration.hook';
import { Sinks } from './sinks/sinks.component';
import { Sources } from './sources/sources.component';

const ITEMS = [
    {
        label: 'Input',
        content: <Sources />,
    },
    {
        label: 'Output',
        content: <Sinks />,
    },
];

type SourcesSinksTabProps = (typeof ITEMS)[number];

const SourcesSinksTabs = () => {
    return (
        <Tabs aria-label={'Sources and sinks tabs'} items={ITEMS}>
            <TabList>{(tab: SourcesSinksTabProps) => <Item key={tab.label}>{tab.label}</Item>}</TabList>
            <TabPanels marginTop={'size-200'}>
                {(tab: SourcesSinksTabProps) => (
                    <Item key={tab.label}>
                        {/*
                            Note: we prefetch sources and sinks using usePrefetchPipelineConfiguration.
                            This Suspense is just in case the prefetch failed.
                         */}
                        <Suspense
                            fallback={
                                <View padding={'size-200'}>
                                    <Loading mode={'inline'} size={'M'} style={{ height: '100%', width: '100%' }} />
                                </View>
                            }
                        >
                            {tab.content}
                        </Suspense>
                    </Item>
                )}
            </TabPanels>
        </Tabs>
    );
};

export const SourcesSinks = () => {
    usePrefetchPipelineConfiguration();

    return (
        <DialogTrigger type={'popover'} hideArrow placement={'bottom right'}>
            <Button variant={'secondary'}>
                <Flex alignItems={'center'} gap={'size-50'}>
                    <InputOutput />
                    <Text>Pipeline configuration</Text>
                </Flex>
            </Button>
            <Dialog>
                <Content UNSAFE_style={{ scrollbarGutter: 'stable' }}>
                    <SourcesSinksTabs />
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
