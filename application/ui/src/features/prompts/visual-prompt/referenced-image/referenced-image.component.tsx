/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, View } from '@geti/ui';
import { Delete } from '@geti/ui/icons';

import styles from './referenced-image.module.scss';

interface ReferencedImageProps {
    image: string;
}

export const ReferencedImage = ({ image }: ReferencedImageProps) => {
    return (
        <View UNSAFE_className={styles.referencedImage}>
            <img src={image} alt={image.toString()} className={styles.image} />
            <View
                position={'absolute'}
                right={'size-100'}
                top={'size-100'}
                backgroundColor={'gray-50'}
                UNSAFE_className={styles.actionMenu}
            >
                <ActionButton isQuiet>
                    <Delete />
                </ActionButton>
            </View>
        </View>
    );
};
