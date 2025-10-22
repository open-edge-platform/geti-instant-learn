/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import createFetchClient from 'openapi-fetch';
import createClient from 'openapi-react-query';

import type { paths } from './openapi-spec';

export const API_BASE_URL = import.meta.env.PUBLIC_API_BASE_URL || 'http://localhost:9100';

export const client = createFetchClient<paths>({ baseUrl: API_BASE_URL });

export const $api = createClient(client);
