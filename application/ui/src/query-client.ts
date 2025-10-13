/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { type paths } from '@geti-prompt/api';
import { matchQuery, MutationCache, QueryClient } from '@tanstack/react-query';
import type { HttpMethod } from 'openapi-typescript-helpers';

type PathsWithMethod<Paths extends paths, Method extends HttpMethod> = {
    [P in keyof Paths]: Method extends keyof Paths[P] ? P : never;
}[keyof Paths];

export type QueryKey<Paths extends paths, Method extends HttpMethod> = [HttpMethod, PathsWithMethod<Paths, Method>];

type MutationMeta = {
    invalidates?: QueryKey<paths, HttpMethod>[];
};

declare module '@tanstack/react-query' {
    interface Register {
        mutationMeta: MutationMeta;
    }
}

export const queryClient = new QueryClient({
    mutationCache: new MutationCache({
        onSuccess: (_data, _variables, _context, mutation) => {
            queryClient.invalidateQueries({
                predicate: (query) => {
                    return mutation.meta?.invalidates?.some((queryKey) => matchQuery({ queryKey }, query)) ?? true;
                },
            });
        },
    }),
});
