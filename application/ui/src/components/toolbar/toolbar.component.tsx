/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CSSProperties } from 'react';

import { useGetSources } from '@geti-prompt/hooks';
import { Button, Flex, StatusLight, View } from '@geti/ui';

import { SourcesSinksConfiguration } from '../../features/sources-sinks-configuration/sources-sinks-configuration.component';
import { useWebRTCConnection } from '../../features/stream/web-rtc/web-rtc-connection-provider';

const StreamStatus = () => {
    const { data } = useGetSources();
    const { status, stop } = useWebRTCConnection();

    if (data === undefined || data.sources.length === 0) {
        return null;
    }

    switch (status) {
        case 'idle':
            return (
                <Flex
                    gap='size-100'
                    alignItems={'center'}
                    UNSAFE_style={
                        {
                            '--spectrum-gray-visual-color': 'var(--spectrum-global-color-gray-500)',
                        } as CSSProperties
                    }
                >
                    <StatusLight role={'status'} aria-label='Idle' variant='neutral'>
                        Idle
                    </StatusLight>
                </Flex>
            );
        case 'connecting':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Connecting' variant='info'>
                        Connecting
                    </StatusLight>
                </Flex>
            );
        case 'disconnected':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Disconnected' variant='negative'>
                        Disconnected
                    </StatusLight>
                </Flex>
            );
        case 'failed':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Failed' variant='negative'>
                        Failed
                    </StatusLight>
                </Flex>
            );
        case 'connected':
            return (
                <Flex gap='size-200' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Connected' variant='positive'>
                        Connected
                    </StatusLight>
                    <Button onPress={stop} variant='secondary'>
                        Stop
                    </Button>
                </Flex>
            );
    }
};

export const Toolbar = () => {
    return (
        <View
            gridArea={'toolbar'}
            borderTopWidth={'thin'}
            borderColor={'gray-50'}
            backgroundColor={'gray-100'}
            paddingX={'size-200'}
        >
            <Flex justifyContent={'space-between'} alignItems={'center'} height={'100%'} width={'100%'}>
                <StreamStatus />
                <View marginStart={'auto'}>
                    <SourcesSinksConfiguration />
                </View>
            </Flex>
        </View>
    );
};
