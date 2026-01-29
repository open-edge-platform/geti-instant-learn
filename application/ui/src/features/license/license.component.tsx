/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, ButtonGroup, Content, Divider, Flex, Heading, Link, Text, View } from '@geti/ui';

import { Layout } from '../project/projects-list-entry/layout.component';

type LicenseProps = {
    onAccept: () => void;
};

export const License = ({ onAccept }: LicenseProps) => {
    return (
        <Layout>
            <Flex justifyContent={'center'} alignItems={'center'} height={'100%'}>
                <View backgroundColor={'gray-100'} padding={'size-400'} borderRadius={'regular'}>
                    <Heading>License</Heading>
                    <Divider marginY={'size-200'} size={'S'} />
                    <Content>
                        <Flex direction={'column'}>
                            <Text>
                                This software is subject to additional third-party licenses. By using it, you agree to:
                            </Text>
                            <ul>
                                <li>
                                    <Link
                                        href={'https://github.com/facebookresearch/sam3/blob/main/LICENSE'}
                                        target={'_blank'}
                                        rel={'noopener noreferrer'}
                                    >
                                        SAM3 License Agreement
                                    </Link>
                                </li>
                                <li>
                                    <Link
                                        href={'https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md'}
                                        target={'_blank'}
                                        rel={'noopener noreferrer'}
                                    >
                                        DINOv3 License Agreement
                                    </Link>
                                </li>
                            </ul>

                            <Text>By using the application I acknowledge I have:</Text>
                            <ul>
                                <li>read and understood the license terms at the links above;</li>
                                <li>confirmed the linked terms govern the contents I seek to access and use; </li>
                                <li>and - accepted and agreed to the linked license terms.</li>
                            </ul>
                        </Flex>
                    </Content>
                    <ButtonGroup width={'100%'} marginTop={'size-200'}>
                        <Button marginStart={'auto'} onPress={onAccept}>
                            Accept & download models
                        </Button>
                    </ButtonGroup>
                </View>
            </Flex>
        </Layout>
    );
};
