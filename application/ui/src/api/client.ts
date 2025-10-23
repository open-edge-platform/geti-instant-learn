/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import createFetchClient from 'openapi-fetch';
import createClient from 'openapi-react-query';

import type { paths } from './openapi-spec';

export const client = createFetchClient<paths>({ baseUrl: import.meta.env.PUBLIC_API_URL ?? 'http://localhost:9100' });

export const $api = createClient(client);
