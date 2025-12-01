/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { type FrameAPIType } from '@geti-prompt/api';

export type FrameType = Pick<FrameAPIType, 'index'> & { thumbnail: FrameAPIType['thumbnail'] | null };
