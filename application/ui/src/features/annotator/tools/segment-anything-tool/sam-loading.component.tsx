/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Flex, Heading, View } from '@geti/ui';

import IntelBrandedLoadingGif from './intel-loading.webp';

export const IntelBrandedLoading = () => {
    return (
        <img
            src={IntelBrandedLoadingGif}
            // eslint-disable-next-line jsx-a11y/no-noninteractive-element-to-interactive-role
            role='progressbar'
            alt='Loading'
            style={{
                display: 'block',
                height: '100%',
            }}
        />
    );
};

interface SAMLoadingProps {
    isLoading: boolean;
    isError?: boolean;
    errorMessage?: string;
    onRetry?: () => void;
}

export const SAMLoading = ({ isLoading, isError = false, errorMessage, onRetry }: SAMLoadingProps) => {
    return (
        <View
            position={'absolute'}
            left={0}
            top={0}
            right={0}
            bottom={0}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-alias-background-color-modal-overlay)',
                zIndex: 10,
            }}
        >
            <Flex alignItems={'center'} justifyContent={'center'} height={'100%'}>
                <View width={'80%'} height={'80%'}>
                    <Flex
                        direction={'column'}
                        alignItems={'center'}
                        justifyContent={'center'}
                        height={'100%'}
                        gap='size-100'
                    >
                        <View flex={1} minHeight={0}>
                            <IntelBrandedLoading />
                        </View>
                        <Heading
                            level={1}
                            UNSAFE_style={{
                                fontSize: 'calc(var(--spectrum-global-dimension-size-200) / var(--zoom-scale, 1))',
                                textShadow: '1px 1px 2px black, 1px 1px 2px white',
                            }}
                        >
                            {isLoading && 'Processing image, please wait...'}
                            {isError && 'Failed to process image.'}
                        </Heading>
                        {isError && errorMessage && (
                            <Heading
                                level={3}
                                UNSAFE_style={{
                                    fontSize: 'calc(var(--spectrum-global-dimension-size-150) / var(--zoom-scale, 1))',
                                    textShadow: '1px 1px 2px black',
                                    opacity: 0.8,
                                    maxWidth: '80%',
                                    textAlign: 'center',
                                    wordBreak: 'break-word',
                                }}
                            >
                                {errorMessage}
                            </Heading>
                        )}
                        {isError && onRetry && (
                            <Button variant='primary' onPress={onRetry}>
                                Retry
                            </Button>
                        )}
                    </Flex>
                </View>
            </Flex>
        </View>
    );
};
