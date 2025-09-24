/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { InputOutput } from '@geti-prompt/icons';
import { Button, Content, Dialog, DialogTrigger, Flex, Item, TabList, TabPanels, Tabs, Text } from '@geti/ui';

import { InputConfiguration } from './input-configuration/input-configuration.component';

const ITEMS = [
    {
        label: 'Input Setup',
        content: <InputConfiguration />,
    },
];

type InputOutputTabProps = (typeof ITEMS)[number];

const InputOutputTabs = () => {
    return (
        <Tabs aria-label={'Input and output configuration tabs'} items={ITEMS}>
            <TabList>{(tab: InputOutputTabProps) => <Item key={tab.label}>{tab.label}</Item>}</TabList>
            <TabPanels marginTop={'size-200'}>
                {(tab: InputOutputTabProps) => <Item key={tab.label}>{tab.content}</Item>}
            </TabPanels>
        </Tabs>
    );
};

export const InputOutputConfiguration = () => {
    return (
        <DialogTrigger type={'popover'} hideArrow placement={'bottom right'}>
            <Button variant={'secondary'} style={'fill'}>
                <Flex alignItems={'center'} gap={'size-50'}>
                    <InputOutput />
                    <Text>Input/Output Setup</Text>
                </Flex>
            </Button>
            {(_close) => (
                <Dialog>
                    <Content>
                        <InputOutputTabs />
                    </Content>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
