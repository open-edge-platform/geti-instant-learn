/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { InputOutput } from '@geti-prompt/icons';
import { Button, Content, Dialog, DialogTrigger, Flex, Item, TabList, TabPanels, Tabs, Text } from '@geti/ui';

import { Sources } from './sources/sources.component';

const ITEMS = [
    {
        label: 'Input Setup',
        content: <Sources />,
    },
];

type SourcesSinksTabProps = (typeof ITEMS)[number];

const SourcesSinksTabs = () => {
    return (
        <Tabs aria-label={'Sources and sinks tabs'} items={ITEMS}>
            <TabList>{(tab: SourcesSinksTabProps) => <Item key={tab.label}>{tab.label}</Item>}</TabList>
            <TabPanels marginTop={'size-200'}>
                {(tab: SourcesSinksTabProps) => <Item key={tab.label}>{tab.content}</Item>}
            </TabPanels>
        </Tabs>
    );
};

export const SourcesSinks = () => {
    return (
        <DialogTrigger type={'popover'} hideArrow placement={'bottom right'}>
            <Button variant={'secondary'}>
                <Flex alignItems={'center'} gap={'size-50'}>
                    <InputOutput />
                    <Text>Pipeline configuration</Text>
                </Flex>
            </Button>
            <Dialog>
                <Content>
                    <SourcesSinksTabs />
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
