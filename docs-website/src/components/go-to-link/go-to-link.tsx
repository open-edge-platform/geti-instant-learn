import Link from '@docusaurus/Link';
import ThemedImage from '@theme/ThemedImage';

import styles from './styles.module.css';

interface GoToLinkProps {
    link: string;
    name: string;
}

export const GoToLink = ({ link, name }: GoToLinkProps) => {
    return (
        <Link to={link} className={styles.goToLink}>
            {name}
            <ThemedImage
                alt={''}
                sources={{
                    light: '/img/light/chevron-right-light.svg',
                    dark: '/img/dark/chevron-right-dark.svg',
                }}
            />
        </Link>
    );
};
