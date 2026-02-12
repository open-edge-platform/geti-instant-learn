import Link from '@docusaurus/Link';
import GithubIcon from '@theme/Icon/Socials/GitHub';

import styles from './styles.module.css';

export const DocCard = ({
    title,
    description,
    docHref,
    githubHref,
}: {
    title: string;
    description: string;
    docHref: string;
    githubHref: string;
}) => {
    return (
        <div className={styles.card}>
            <div className={styles.cardHeader}>
                <Link to={docHref} className={styles.cardLink}>
                    <h2 className={styles.cardTitle}>📄️ {title}</h2>
                </Link>
                <Link to={githubHref} className={styles.cardGithubLink}>
                    <GithubIcon height={24} width={24} />
                </Link>
            </div>
            <Link to={docHref} className={styles.cardLink}>
                <p className={styles.cardDescription}>{description}</p>
            </Link>
        </div>
    );
};
