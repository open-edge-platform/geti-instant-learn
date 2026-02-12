import { DEEP_LEARNING_COMPONENTS } from '@site/config';

import { DocCard } from './doc-card';

import styles from './styles.module.css';

export const DocCardList = () => {
    return (
        <div className={styles.cardList}>
            {DEEP_LEARNING_COMPONENTS.map((component) => {
                return (
                    <DocCard
                        title={component.title}
                        description={component.description}
                        docHref={component.docHref}
                        githubHref={component.githubHref}
                    />
                );
            })}
        </div>
    );
};
