/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { Button, Flex } from '@geti/ui';
import { Add as AddIcon } from '@geti/ui/icons';

import styles from './existing-pipeline-entities.module.scss';

interface ExistingPipelineEntitiesListProps {
    children: ReactNode;
}

const ExistingPipelineEntitiesList = ({ children }: ExistingPipelineEntitiesListProps) => {
    return (
        <Flex direction={'column'} gap={'size-100'}>
            {children}
        </Flex>
    );
};

interface AddNewEntityProps {
    onPress: () => void;
    text: string;
}

const AddNewEntity = ({ onPress, text }: AddNewEntityProps) => {
    return (
        <Button variant={'secondary'} onPress={onPress} UNSAFE_className={styles.addNewButton}>
            <AddIcon /> {text}
        </Button>
    );
};

interface ExistingPipelineEntitiesProps {
    addNewEntityButton: ReactNode;
    children: ReactNode;
}

export const ExistingPipelineEntities = ({ addNewEntityButton, children }: ExistingPipelineEntitiesProps) => {
    return (
        <Flex direction={'column'} gap={'size-200'}>
            {addNewEntityButton}
            {children}
        </Flex>
    );
};

ExistingPipelineEntities.AddNewEntityButton = AddNewEntity;
ExistingPipelineEntities.List = ExistingPipelineEntitiesList;
