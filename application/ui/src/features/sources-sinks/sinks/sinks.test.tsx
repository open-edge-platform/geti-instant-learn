/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SinksListType, SinkUpdateType } from '@/api';
import { getMockedMQTTSink, render } from '@/test-utils';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http, server } from '../../../setup-test';
import { Sinks } from './sinks.component';

const mockSinksResponse = (sinks: SinksListType['sinks']): SinksListType => ({
    sinks,
    pagination: { count: sinks.length, total: sinks.length, offset: 0, limit: 10 },
});

const openCardMenu = async (cardTestId = 'pipeline-entity-card-mqtt') => {
    const card = await screen.findByTestId(cardTestId);
    await userEvent.click(within(card).getByRole('button'));
};

describe('Sinks', () => {
    describe('Empty state', () => {
        it('renders MQTT add option when no sinks exist', async () => {
            render(<Sinks />);

            expect(await screen.findByRole('button', { name: 'MQTT' })).toBeInTheDocument();
        });

        it('does not show Add new sink button when MQTT already exists', async () => {
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([getMockedMQTTSink()]));
                })
            );

            render(<Sinks />);

            await screen.findByTestId('pipeline-entity-card-mqtt');

            expect(screen.queryByRole('button', { name: 'Add new sink' })).not.toBeInTheDocument();
        });
    });

    describe('Create MQTT sink', () => {
        it('submits create request with all values including defaults and navigates to existing view', async () => {
            let sinkCreated = false;
            server.use(
                http.post('/api/v1/projects/{project_id}/sinks', async ({ request }) => {
                    sinkCreated = true;
                    const body = await request.json();
                    expect(body).toMatchObject({
                        active: true,
                        config: {
                            sink_type: 'mqtt',
                            name: 'My Sink',
                            broker_host: 'broker.local',
                            topic: 'sensors/data',
                            broker_port: 1883,
                            auth_required: true,
                        },
                    });
                    return HttpResponse.json(getMockedMQTTSink(), { status: 201 });
                }),
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse(sinkCreated ? [getMockedMQTTSink()] : []));
                })
            );

            render(<Sinks />);

            await userEvent.type(await screen.findByRole('textbox', { name: /name/i }), 'My Sink');
            await userEvent.type(screen.getByRole('textbox', { name: /broker host/i }), 'broker.local');
            await userEvent.type(screen.getByRole('textbox', { name: /topic/i }), 'sensors/data');

            await userEvent.click(screen.getByRole('button', { name: 'Apply' }));

            await waitFor(() => {
                expect(screen.getByTestId('pipeline-entity-card-mqtt')).toBeInTheDocument();
            });
        });

        it('does not submit create request when required fields are empty', async () => {
            render(<Sinks />);

            const nameField = await screen.findByRole('textbox', { name: /name/i });
            const brokerHostField = screen.getByRole('textbox', { name: /broker host/i });
            const topicField = screen.getByRole('textbox', { name: /topic/i });
            const applyButton = screen.getByRole('button', { name: 'Apply' });

            expect(nameField).toHaveValue('');
            expect(brokerHostField).toHaveValue('');
            expect(topicField).toHaveValue('');
            expect(applyButton).toBeDisabled();

            await userEvent.type(nameField, 'My Sink');
            expect(applyButton).toBeDisabled();

            await userEvent.type(brokerHostField, 'localhost');
            expect(applyButton).toBeDisabled();

            await userEvent.type(topicField, 'my/topic');
            expect(applyButton).toBeEnabled();
        });
    });

    describe('Existing MQTT sink', () => {
        it('renders summary card with sink details', async () => {
            const sink = getMockedMQTTSink({
                config: {
                    sink_type: 'mqtt',
                    name: 'Test Sink',
                    broker_host: 'my-broker.local',
                    topic: 'events/all',
                    broker_port: 8883,
                    auth_required: false,
                },
            });
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([sink]));
                })
            );

            render(<Sinks />);

            const card = await screen.findByTestId('pipeline-entity-card-mqtt');
            expect(within(card).getByText('Name: Test Sink')).toBeInTheDocument();
            expect(within(card).getByText('Broker host: my-broker.local')).toBeInTheDocument();
            expect(within(card).getByText('Topic: events/all')).toBeInTheDocument();
            expect(within(card).getByText('Broker port: 8883')).toBeInTheDocument();
            expect(within(card).getByText('Auth required: No')).toBeInTheDocument();
        });

        it('active sink menu shows Edit and Delete but not Connect', async () => {
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([getMockedMQTTSink({ active: true })]));
                })
            );

            render(<Sinks />);

            await openCardMenu();

            expect(screen.getByRole('menuitem', { name: 'Edit' })).toBeInTheDocument();
            expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
            expect(screen.queryByRole('menuitem', { name: 'Connect' })).not.toBeInTheDocument();
        });

        it('inactive sink menu shows Edit, Delete, and Connect', async () => {
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([getMockedMQTTSink({ active: false })]));
                })
            );

            render(<Sinks />);

            await openCardMenu();

            expect(screen.getByRole('menuitem', { name: 'Edit' })).toBeInTheDocument();
            expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
            expect(screen.getByRole('menuitem', { name: 'Connect' })).toBeInTheDocument();
        });
    });

    describe('Connect inactive sink', () => {
        it('sends update request with active: true and preserves config', async () => {
            const sink = getMockedMQTTSink({ id: 'sink-123', active: false });
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([sink]));
                })
            );

            render(<Sinks />);

            let updateBody: SinkUpdateType | null = null;
            let updatedSinkId: string | undefined;
            server.use(
                http.put('/api/v1/projects/{project_id}/sinks/{sink_id}', async ({ request, params }) => {
                    updatedSinkId = params.sink_id;
                    updateBody = await request.json();
                    return HttpResponse.json({ ...sink, active: true });
                })
            );

            await openCardMenu();
            await userEvent.click(screen.getByRole('menuitem', { name: 'Connect' }));

            await waitFor(() => {
                expect(updatedSinkId).toBe('sink-123');
                expect(updateBody).toMatchObject({
                    active: true,
                    config: sink.config,
                });
            });
        });
    });

    describe('Edit active MQTT sink', () => {
        it('prefills fields with current values', async () => {
            const sink = getMockedMQTTSink({
                active: true,
                config: {
                    sink_type: 'mqtt',
                    name: 'Existing Sink',
                    broker_host: 'existing-host',
                    topic: 'existing/topic',
                    broker_port: 1883,
                    auth_required: true,
                },
            });
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([sink]));
                })
            );

            render(<Sinks />);

            await openCardMenu();
            await userEvent.click(screen.getByRole('menuitem', { name: 'Edit' }));

            expect(screen.getByRole('textbox', { name: /name/i })).toHaveValue('Existing Sink');
            expect(screen.getByRole('textbox', { name: /broker host/i })).toHaveValue('existing-host');
            expect(screen.getByRole('textbox', { name: /topic/i })).toHaveValue('existing/topic');
            expect(screen.getByLabelText(/broker port/i, { selector: 'input' })).toHaveValue('1,883');
            expect(screen.getByRole('switch', { name: /auth required/i })).toBeChecked();
        });

        it('sends update request with active: true when Save is clicked', async () => {
            const sink = getMockedMQTTSink({
                id: 'sink-abc',
                active: true,
                config: {
                    sink_type: 'mqtt',
                    name: 'Old Name',
                    broker_host: 'host',
                    topic: 'topic',
                    broker_port: 1883,
                    auth_required: true,
                },
            });
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([sink]));
                })
            );

            render(<Sinks />);

            let updateBody: SinkUpdateType | null = null;
            server.use(
                http.put('/api/v1/projects/{project_id}/sinks/{sink_id}', async ({ request }) => {
                    updateBody = await request.json();
                    return HttpResponse.json({ ...sink, config: { ...sink.config, name: 'New Name' } });
                })
            );

            await openCardMenu();
            await userEvent.click(screen.getByRole('menuitem', { name: 'Edit' }));

            const nameField = screen.getByRole('textbox', { name: /name/i });
            await userEvent.clear(nameField);
            await userEvent.type(nameField, 'New Name');

            await userEvent.click(screen.getByRole('button', { name: 'Save' }));

            await waitFor(() => {
                expect(updateBody).toMatchObject({
                    active: true,
                    config: expect.objectContaining({ name: 'New Name' }),
                });
            });
        });

        it('does not show Save & Connect button for active sink', async () => {
            const sink = getMockedMQTTSink({ active: true });
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([sink]));
                })
            );

            render(<Sinks />);

            await openCardMenu();
            await userEvent.click(screen.getByRole('menuitem', { name: 'Edit' }));

            expect(screen.queryByRole('button', { name: 'Save & Connect' })).not.toBeInTheDocument();
        });
    });

    describe('Edit inactive MQTT sink', () => {
        it('shows Save & Connect button, disabled until config changes', async () => {
            const sink = getMockedMQTTSink({
                active: false,
                config: {
                    sink_type: 'mqtt',
                    name: 'Old Name',
                    broker_host: 'host',
                    topic: 'topic',
                    broker_port: 1883,
                    auth_required: true,
                },
            });
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([sink]));
                })
            );

            render(<Sinks />);

            await openCardMenu();
            await userEvent.click(screen.getByRole('menuitem', { name: 'Edit' }));

            expect(screen.getByRole('button', { name: 'Save & Connect' })).toBeInTheDocument();

            expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
            expect(screen.getByRole('button', { name: 'Save & Connect' })).toBeDisabled();

            const nameField = screen.getByRole('textbox', { name: /name/i });
            await userEvent.clear(nameField);
            await userEvent.type(nameField, 'New Name');

            expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled();
            expect(screen.getByRole('button', { name: 'Save & Connect' })).toBeEnabled();
        });

        it('Save keeps sink inactive', async () => {
            const sink = getMockedMQTTSink({
                id: 'sink-xyz',
                active: false,
                config: {
                    sink_type: 'mqtt',
                    name: 'Old Name',
                    broker_host: 'host',
                    topic: 'topic',
                    broker_port: 1883,
                    auth_required: true,
                },
            });
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([sink]));
                })
            );

            render(<Sinks />);

            let updateBody: SinkUpdateType | null = null;
            server.use(
                http.put('/api/v1/projects/{project_id}/sinks/{sink_id}', async ({ request }) => {
                    updateBody = await request.json();
                    return HttpResponse.json(sink);
                })
            );

            await openCardMenu();
            await userEvent.click(screen.getByRole('menuitem', { name: 'Edit' }));

            const nameField = screen.getByRole('textbox', { name: /name/i });
            await userEvent.clear(nameField);
            await userEvent.type(nameField, 'New Name');

            await userEvent.click(screen.getByRole('button', { name: 'Save' }));

            await waitFor(() => {
                expect(updateBody).toMatchObject({
                    active: false,
                    config: expect.objectContaining({ name: 'New Name' }),
                });
            });
        });

        it('Save & Connect activates the inactive sink while saving config changes', async () => {
            const sink = getMockedMQTTSink({
                id: 'sink-xyz',
                active: false,
                config: {
                    sink_type: 'mqtt',
                    name: 'Old Name',
                    broker_host: 'host',
                    topic: 'topic',
                    broker_port: 1883,
                    auth_required: true,
                },
            });
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([sink]));
                })
            );

            render(<Sinks />);

            let updateBody: SinkUpdateType | null = null;
            server.use(
                http.put('/api/v1/projects/{project_id}/sinks/{sink_id}', async ({ request }) => {
                    updateBody = await request.json();
                    return HttpResponse.json({ ...sink, active: true });
                })
            );

            await openCardMenu();
            await userEvent.click(screen.getByRole('menuitem', { name: 'Edit' }));

            const nameField = screen.getByRole('textbox', { name: /name/i });
            await userEvent.clear(nameField);
            await userEvent.type(nameField, 'New Name');

            await userEvent.click(screen.getByRole('button', { name: 'Save & Connect' }));

            await waitFor(() => {
                expect(updateBody).toMatchObject({ active: true });
            });
        });
    });

    describe('Delete sink', () => {
        it('sends delete request with correct sink id and returns to list view', async () => {
            const sink = getMockedMQTTSink({ id: 'sink-to-delete', active: true });
            server.use(
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([sink]));
                })
            );

            render(<Sinks />);

            let deletedSinkId: string | undefined;
            server.use(
                http.delete('/api/v1/projects/{project_id}/sinks/{sink_id}', ({ params, response }) => {
                    deletedSinkId = params.sink_id as string;
                    return response(204).empty();
                }),
                http.get('/api/v1/projects/{project_id}/sinks', ({ response }) => {
                    return response(200).json(mockSinksResponse([]));
                })
            );

            await openCardMenu();
            await userEvent.click(screen.getByRole('menuitem', { name: 'Delete' }));

            await waitFor(() => {
                expect(deletedSinkId).toBe('sink-to-delete');
            });

            await waitFor(() => {
                expect(screen.getByRole('button', { name: 'MQTT' })).toBeInTheDocument();
            });
        });
    });
});
