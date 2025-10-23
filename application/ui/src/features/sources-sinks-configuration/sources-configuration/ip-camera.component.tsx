/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { Button, Flex, Form, Item, Picker, TextField } from '@geti/ui';
import { isNull } from 'lodash-es';

export const IPCameraForm = () => {
    const protocols = ['RTSP', 'HTTP', 'HTTPS', 'ONVIF', 'SMTP', 'TCP'];

    const [ip, setIp] = useState('');
    const [port, setPort] = useState('');
    const [streamPath, setStreamPath] = useState('');
    const [selectedProtocol, setSelectedProtocol] = useState(protocols[0]);

    const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
    };

    const isFormValid = ip && port && streamPath && selectedProtocol;

    return (
        <Form onSubmit={handleSubmit}>
            <Flex direction='row' gap='size-200'>
                <TextField label='IP Address' value={ip} onChange={setIp} />
                <TextField label='Port' value={port} onChange={setPort} />
            </Flex>
            <TextField label='Stream Path' value={streamPath} onChange={setStreamPath} />
            <Picker
                label='Protocol'
                defaultSelectedKey={selectedProtocol}
                items={protocols}
                onSelectionChange={(key) => !isNull(key) && setSelectedProtocol(key as string)}
            >
                {protocols.map((protocol) => (
                    <Item key={protocol}>{protocol}</Item>
                ))}
            </Picker>
            <Button type='submit' isDisabled={!isFormValid} width={'fit-content'}>
                Apply
            </Button>
        </Form>
    );
};
