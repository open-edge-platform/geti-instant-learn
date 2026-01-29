/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Flex, Heading, Link, Text } from '@geti/ui';

import { Layout } from '../project/projects-list-entry/layout.component';

type LicenseProps = {
    isVisible: boolean;
    onAccept: () => void;
};

export const License = ({ isVisible, onAccept }: LicenseProps) => {
    return (
        <Layout>
            <DialogContainer onDismiss={() => {}} isDismissable={false} isKeyboardDismissDisabled={false}>
                {isVisible && (
                    <Dialog>
                        <Heading>License</Heading>
                        <Divider />
                        <Content>
                            <Flex direction={'column'}>
                                <Text>
                                    This software is subject to additional third-party licenses. By using it, you agree
                                    to:
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

                                <Text>By using the library I acknowledge I have:</Text>
                                <ul>
                                    <li>read and understood the license terms at the links above;</li>
                                    <li>confirmed the linked terms govern the contents I seek to access and use; </li>
                                    <li>and - accepted and agreed to the linked license terms.</li>
                                </ul>
                            </Flex>
                        </Content>
                        <ButtonGroup>
                            <Button onPress={onAccept}>Accept & download models</Button>
                        </ButtonGroup>
                    </Dialog>
                )}
            </DialogContainer>
        </Layout>
    );
};
