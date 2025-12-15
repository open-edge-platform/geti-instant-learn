/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { RefObject } from 'react';

import {
    ActionButton,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    DOMRefValue,
    Flex,
    Heading,
    Item,
    TabList,
    TabPanels,
    Tabs,
    Text,
    Tooltip,
    TooltipTrigger,
    useUnwrapDOMRef,
} from '@geti/ui';
import { Adjustments, Close } from '@geti/ui/icons';

import { useFullScreenMode } from '../full-screen-mode.component';
import { CanvasSettings } from './canvas-settings.component';
import { Hotkeys } from './hotkeys.component';

import styles from './settings.module.scss';

interface SettingsProps {
    ref: RefObject<DOMRefValue<HTMLDivElement> | null>;
}

const tabs = [{ label: 'Canvas settings' }, { label: 'Hotkeys' }];

const SettingsTabs = () => {
    return (
        <Tabs items={tabs}>
            <TabList marginBottom={'size-200'}>
                {(tab: { label: string }) => <Item key={tab.label}>{tab.label}</Item>}
            </TabList>
            <TabPanels>
                <Item key={'Canvas settings'}>
                    <CanvasSettings />
                </Item>
                <Item key={'Hotkeys'}>
                    <Hotkeys />
                </Item>
            </TabPanels>
        </Tabs>
    );
};

export const Settings = ({ ref }: SettingsProps) => {
    const targetRef = useUnwrapDOMRef(ref);
    const { isFullScreenMode } = useFullScreenMode();

    return (
        <DialogTrigger
            type={'popover'}
            hideArrow
            targetRef={targetRef}
            placement={isFullScreenMode ? 'top right' : 'right'}
        >
            <TooltipTrigger>
                <ActionButton isQuiet aria-label={'Settings'}>
                    <Adjustments />
                </ActionButton>
                <Tooltip>Settings</Tooltip>
            </TooltipTrigger>
            {(close) => (
                <Dialog height={'40rem'} UNSAFE_className={styles.settingsDialog}>
                    <Heading>
                        <Flex justifyContent={'space-between'} alignItems={'center'}>
                            <Text>Settings</Text>
                            <ActionButton isQuiet onPress={close} aria-label={'Close settings'}>
                                <Close />
                            </ActionButton>
                        </Flex>
                    </Heading>
                    <Divider size={'S'} />
                    <Content>
                        <SettingsTabs />
                    </Content>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
