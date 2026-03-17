/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Content, Divider, Flex, Heading, Link, Text, View } from '@geti/ui';

import { Layout } from '../project/projects-list-entry/layout.component';

type LicenseProps = {
    onAccept: () => void;
    isAccepting?: boolean;
};

export const License = ({ onAccept, isAccepting = false }: LicenseProps) => {
    return (
        <Layout>
            <Flex justifyContent={'center'} alignItems={'center'} height={'100%'}>
                <View
                    backgroundColor={'gray-50'}
                    padding={'size-400'}
                    borderRadius={'regular'}
                    maxWidth={'size-6000'}
                    width={'100%'}
                >
                    <Heading level={2}>License Agreement</Heading>
                    <Divider marginY={'size-200'} size={'S'} />
                    <Content>
                        <Flex direction={'column'} gap={'size-200'}>
                            <Text>By installing, using, or distributing this application, you acknowledge that:</Text>
                            <ul>
                                <li>you have read and understood the license terms at the links below;</li>
                                <li>confirmed the linked terms govern the contents you seek to access and use; and</li>
                                <li>accepted and agreed to the linked license terms.</li>
                            </ul>
                            <Text>License links</Text>
                            <ul>
                                <li>
                                    <Link
                                        href={'https://github.com/facebookresearch/sam3/blob/main/LICENSE'}
                                        target={'_blank'}
                                        rel={'noopener noreferrer'}
                                    >
                                        SAM3 License
                                    </Link>
                                </li>
                                <li>
                                    <Link
                                        href={'https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md'}
                                        target={'_blank'}
                                        rel={'noopener noreferrer'}
                                    >
                                        DINOv3 License
                                    </Link>
                                </li>
                            </ul>
                        </Flex>
                    </Content>
                    <Flex justifyContent={'end'} marginTop={'size-300'}>
                        <Button variant={'accent'} onPress={onAccept} isPending={isAccepting}>
                            Accept and continue
                        </Button>
                    </Flex>
                </View>
            </Flex>
        </Layout>
    );
};
