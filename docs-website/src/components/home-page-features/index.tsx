import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import ThemedImage from '@theme/ThemedImage';

import styles from './styles.module.css';

type FeatureItem = {
    title: string;
    description: JSX.Element;
    link: string;
    icon: { alt: string; sources: { light: string; dark: string } };
};

const FeatureList: FeatureItem[] = [
    {
        title: 'Free & Open Access',
        link: 'https://github.com/open-edge-platform/geti',
        icon: {
            alt: 'Free deployment options: install on your hardware or access the source code on GitHub.',
            sources: {
                light: '/img/light/github-icon-light.svg',
                dark: '/img/dark/github-icon-dark.svg',
            },
        },
        description: (
            <>
                Geti™ is part of Intel's Open Edge Platform, an open source ecosystem that provides scalable edge solutions.
                Use Geti™ the way you want: deploy the software for free on your own hardware with our
                installer, or access the source code on GitHub.
            </>
        ),
    },
];

function Feature({ title, description, link, icon: { alt, sources } }: FeatureItem) {
    return (
        <div className={styles.featureItem}>
            <div className={styles.featureImage}>
                <ThemedImage alt={alt} sources={sources} />
            </div>

            <div className={styles.featureDescription}>
                <div>
                    <Heading as='h4' className={styles.featureTitle}>
                        {title}
                    </Heading>
                    <p>{description}</p>
                </div>
                <Link href={link}>Read more</Link>
            </div>
        </div>
    );
}

export default function HomepageFeatures(): JSX.Element {
    return (
        <section className={styles.featuresList}>
            {FeatureList.map((props, idx) => (
                <Feature key={idx} {...props} />
            ))}
        </section>
    );
}
