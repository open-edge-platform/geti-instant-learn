/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { paths } from '@/api';
import { toast } from '@geti/ui';
import { matchQuery, MutationCache, Query, QueryCache, QueryClient } from '@tanstack/react-query';

import type { Meta, QueryKey } from './query-client.interface';

declare module '@tanstack/react-query' {
    interface Register {
        mutationMeta: Meta;
        queryMeta: Meta;
    }
}

export const getQueryKey = (queryKey: QueryKey<paths>): QueryKey<paths> => {
    return queryKey;
};

const TOAST_DURATION = 5000;

const getErrorMessage = (error: unknown, customMessage?: string): string => {
    if (customMessage) {
        return customMessage;
    }

    if (error && typeof error === 'object') {
        if ('detail' in error && typeof error.detail === 'string') {
            return error.detail;
        }

        if ('message' in error && typeof error.message === 'string') {
            if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
                return 'Network error. Please check your connection and try again.';
            }

            return error.message;
        }
    }

    return 'An unexpected error occurred. Please try again.';
};

export const queryClient: QueryClient = new QueryClient({
    defaultOptions: {
        queries: {
            retry: false,
        },
    },
    queryCache: new QueryCache({
        onError: (error, query) => {
            if (query.meta?.error?.notify) {
                const errorMessage = query.meta?.error?.message;

                toast({
                    type: 'error',
                    message: getErrorMessage(error, errorMessage),
                    duration: TOAST_DURATION,
                });
            }
        },
    }),
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
        onError: (error, _variables, _context, mutation) => {
            if (mutation.meta?.error?.notify) {
                const errorMessage = mutation.meta?.error?.message;

                toast({
                    type: 'error',
                    message: getErrorMessage(error, errorMessage),
                    duration: TOAST_DURATION,
                });
            }
        },
    }),
});
