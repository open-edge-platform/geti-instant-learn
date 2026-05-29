/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { MQTTSinkType } from '@/api';
import { expect, http, test } from '@/test-fixtures';

import { ACTIVE_MQTT_SINK, INACTIVE_MQTT_SINK, mockSinksResponse } from './mocks';

test.describe('Sinks', () => {
    test('Creates an MQTT sink and shows it in the existing sinks view', async ({ network, sinksPage }) => {
        let sinks: MQTTSinkType[] = [];

        network.use(
            http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                return response(200).json(mockSinksResponse(sinks));
            }),
            http.post('/api/v1/projects/{project_id}/sinks', async ({ request, response }) => {
                const body = await request.json();

                expect(body).toMatchObject({
                    active: true,
                    config: {
                        sink_type: 'mqtt',
                        name: 'Test Sink',
                        broker_host: 'broker.local',
                        topic: 'sensors/data',
                        broker_port: 1883,
                        auth_required: true,
                    },
                });

                sinks = [ACTIVE_MQTT_SINK];
                return response(201).json(ACTIVE_MQTT_SINK);
            })
        );

        await sinksPage.goto();
        await sinksPage.openPipelineConfiguration();
        await sinksPage.openOutputTab();

        await test.step('Fill in MQTT fields and submit', async () => {
            await sinksPage.nameField.fill('Test Sink');
            await sinksPage.brokerHostField.fill('broker.local');
            await sinksPage.topicField.fill('sensors/data');
            await sinksPage.applyButton.click();
        });

        await test.step('Created MQTT sink card appears', async () => {
            await expect(sinksPage.getSinkCard()).toBeVisible();
        });
    });

    test('Manages an existing active MQTT sink — displays summary and edits config', async ({ network, sinksPage }) => {
        network.use(
            http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                return response(200).json(mockSinksResponse([ACTIVE_MQTT_SINK]));
            })
        );

        await sinksPage.goto();
        await sinksPage.openPipelineConfiguration();
        await sinksPage.openOutputTab();

        await test.step('Card displays all MQTT summary fields', async () => {
            const card = sinksPage.getSinkCard();
            await expect(card.getByText(`Name: ${ACTIVE_MQTT_SINK.config.name}`)).toBeVisible();
            await expect(card.getByText(`Broker host: ${ACTIVE_MQTT_SINK.config.broker_host}`)).toBeVisible();
            await expect(card.getByText(`Topic: ${ACTIVE_MQTT_SINK.config.topic}`)).toBeVisible();
            await expect(card.getByText(`Broker port: ${ACTIVE_MQTT_SINK.config.broker_port}`)).toBeVisible();
            await expect(card.getByText('Auth required: Yes')).toBeVisible();
        });

        await test.step('Active sink menu has Edit and Delete but not Connect', async () => {
            await sinksPage.openSinkActions();
            await expect(sinksPage.editMenuItem).toBeVisible();
            await expect(sinksPage.deleteMenuItem).toBeVisible();
            await expect(sinksPage.connectMenuItem).toBeHidden();
        });

        await test.step('Edit form is prefilled with current values', async () => {
            await sinksPage.selectAction('Edit');
            await expect(sinksPage.nameField).toHaveValue(ACTIVE_MQTT_SINK.config.name);
            await expect(sinksPage.brokerHostField).toHaveValue(ACTIVE_MQTT_SINK.config.broker_host);
            await expect(sinksPage.topicField).toHaveValue(ACTIVE_MQTT_SINK.config.topic);
        });

        await test.step('Save sends updated config and preserves active: true', async () => {
            let updateBody: Record<string, unknown> | null = null;
            network.use(
                http.put('/api/v1/projects/{project_id}/sinks/{sink_id}', async ({ request, response }) => {
                    updateBody = await request.json();
                    return response(200).json(ACTIVE_MQTT_SINK);
                })
            );

            await sinksPage.brokerHostField.fill('updated-broker.local');
            await sinksPage.saveButton.click();

            expect(updateBody).toMatchObject({
                active: true,
                config: expect.objectContaining({ broker_host: 'updated-broker.local' }),
            });
        });
    });

    test('Manages an existing inactive MQTT sink — Save & Connect activates it while saving config changes', async ({
        network,
        sinksPage,
    }) => {
        network.use(
            http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                return response(200).json(mockSinksResponse([INACTIVE_MQTT_SINK]));
            })
        );

        await sinksPage.goto();
        await sinksPage.openPipelineConfiguration();
        await sinksPage.openOutputTab();

        await test.step('Inactive sink menu has Edit, Delete, and Connect', async () => {
            await sinksPage.openSinkActions();
            await expect(sinksPage.editMenuItem).toBeVisible();
            await expect(sinksPage.deleteMenuItem).toBeVisible();
            await expect(sinksPage.connectMenuItem).toBeVisible();
        });

        await test.step('Save & Connect is disabled before any config change', async () => {
            await sinksPage.selectAction('Edit');
            await expect(sinksPage.saveAndConnectButton).toBeDisabled();
            await expect(sinksPage.saveButton).toBeDisabled();
        });

        await test.step('Save & Connect activates the inactive sink while saving config changes', async () => {
            let updateBody: Record<string, unknown> | null = null;
            network.use(
                http.put('/api/v1/projects/{project_id}/sinks/{sink_id}', async ({ request, response }) => {
                    updateBody = await request.json();
                    return response(200).json({ ...INACTIVE_MQTT_SINK, active: true });
                })
            );

            await sinksPage.brokerHostField.fill('new-broker.local');
            await sinksPage.saveAndConnectButton.click();

            expect(updateBody).toMatchObject({
                active: true,
                config: expect.objectContaining({ broker_host: 'new-broker.local' }),
            });
        });
    });

    test('Connects an inactive MQTT sink directly from the card menu', async ({ network, sinksPage }) => {
        let sinks = [INACTIVE_MQTT_SINK];

        network.use(
            http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                return response(200).json(mockSinksResponse(sinks));
            })
        );

        await sinksPage.goto();
        await sinksPage.openPipelineConfiguration();
        await sinksPage.openOutputTab();

        let updateBody: Record<string, unknown> | null = null;
        network.use(
            http.put('/api/v1/projects/{project_id}/sinks/{sink_id}', async ({ request, response }) => {
                updateBody = await request.json();
                sinks = [{ ...INACTIVE_MQTT_SINK, active: true }];

                return response(200).json(sinks[0]);
            })
        );

        await sinksPage.openSinkActions();
        await sinksPage.selectAction('Connect');

        expect(updateBody).toMatchObject({
            active: true,
            config: INACTIVE_MQTT_SINK.config,
        });

        await test.step('Connect is no longer available after the sink becomes active', async () => {
            await sinksPage.openSinkActions();
            await expect(sinksPage.connectMenuItem).toBeHidden();
        });
    });

    test('Deletes the only MQTT sink and returns to the MQTT add option', async ({ network, sinksPage }) => {
        let sinks: MQTTSinkType[] = [ACTIVE_MQTT_SINK];

        network.use(
            http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                return response(200).json(mockSinksResponse(sinks));
            }),
            http.delete('/api/v1/projects/{project_id}/sinks/{sink_id}', ({ params, response }) => {
                expect(params.sink_id).toBe(ACTIVE_MQTT_SINK.id);
                sinks = [];
                return response(204).empty();
            })
        );

        await sinksPage.goto();
        await sinksPage.openPipelineConfiguration();
        await sinksPage.openOutputTab();

        await sinksPage.openSinkActions();
        await sinksPage.selectAction('Delete');

        await expect(sinksPage.mqttSinkTypeButton).toBeVisible();
    });
});
