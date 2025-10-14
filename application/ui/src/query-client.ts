/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { type paths } from '@geti-prompt/api';
import { matchQuery, MutationCache, QueryClient, type Query } from '@tanstack/react-query';
import type { HttpMethod } from 'openapi-typescript-helpers';

type OperationFor<Paths extends paths, P extends keyof Paths, Method extends HttpMethod> = Method extends keyof Paths[P]
    ? Paths[P][Method]
    : never;

type PathParamsFor<Paths extends paths, P extends keyof Paths, Method extends HttpMethod> =
    OperationFor<Paths, P, Method> extends { parameters: { path: infer PP } } ? PP : never;

type MethodsForPath<Paths extends paths, P extends keyof Paths> = Extract<keyof Paths[P], HttpMethod>;

export type QueryKey<Paths extends paths> = {
    [P in keyof Paths]: {
        [M in MethodsForPath<Paths, P>]: PathParamsFor<Paths, P, M> extends never
            ? [M, P]
            : [
                  M,
                  P,
                  {
                      params: {
                          path: PathParamsFor<Paths, P, M>;
                      };
                  },
              ];
    }[MethodsForPath<Paths, P>];
}[keyof Paths];

type MutationMeta = {
    invalidates?: QueryKey<paths>[];
    awaits?: QueryKey<paths>[];
};

declare module '@tanstack/react-query' {
    interface Register {
        mutationMeta: MutationMeta;
    }
}

export const queryClient: QueryClient = new QueryClient({
    mutationCache: new MutationCache({
        onSuccess: (_data, _variables, _context, mutation): void | Promise<void> => {
            // Fire-and-forget invalidation
            queryClient.invalidateQueries({
                predicate: (query: Query): boolean => {
                    return (
                        mutation.meta?.invalidates?.some((queryKey) => {
                            return matchQuery({ queryKey }, query);
                        }) ?? false
                    );
                },
            });

            // Optionally await specific query invalidations
            if (mutation.meta?.awaits && mutation.meta.awaits.length > 0) {
                return queryClient.invalidateQueries(
                    {
                        predicate: (query) => {
                            return (
                                mutation.meta?.awaits?.some((queryKey) => matchQuery({ queryKey }, query), {}) ?? false
                            );
                        },
                    },
                    { cancelRefetch: false }
                );
            }
        },
    }),
});
