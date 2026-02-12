import React, { Fragment } from 'react';

import clsx from 'clsx';

import styles from './styles.module.css';

const FOOTER_ITEMS = [
    { name: '©2025 Intel Corporation', href: 'https://www.intel.com/content/www/us/en/homepage.html' },
    { name: 'Terms of Use', href: 'https://www.intel.com/content/www/us/en/legal/terms-of-use.html' },
    { name: 'Cookies', href: 'https://www.intel.com/content/www/us/en/privacy/intel-cookie-notice.html' },
    { name: 'Privacy Policy', href: 'https://www.intel.com/content/www/us/en/privacy/intel-privacy-notice.html' },
];

export default function Footer() {
    return (
        <footer className={clsx('footer', styles.footer)}>
            {FOOTER_ITEMS.map(({ name, href }, index) => (
                <Fragment key={`${index}-${name}`}>
                    <a className={styles.footerItem} href={href}>
                        {name}
                    </a>
                    <hr className={styles.divider} />
                </Fragment>
            ))}
        </footer>
    );
}
