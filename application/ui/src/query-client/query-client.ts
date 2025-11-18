/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { toast } from '@geti/ui';
import { matchQuery, MutationCache, Query, QueryCache, QueryClient } from '@tanstack/react-query';

import { Meta } from './query-client.interface';

declare module '@tanstack/react-query' {
    interface Register {
        mutationMeta: Meta;
        queryMeta: Meta;
    }
}

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
            const errorMessage = query.meta?.errorMessage;

            if (errorMessage) {
                toast({
                    type: 'error',
                    message: getErrorMessage(error, query.meta?.errorMessage),
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
            const errorMessage = mutation.meta?.errorMessage;

            if (errorMessage) {
                toast({
                    type: 'error',
                    message: getErrorMessage(error, mutation.meta?.errorMessage),
                    duration: TOAST_DURATION,
                });
            }
        },
    }),
});
