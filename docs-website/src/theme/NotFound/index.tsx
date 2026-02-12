import React, { type ReactNode } from 'react';

import BrowserOnly from '@docusaurus/BrowserOnly';
import type { WrapperProps } from '@docusaurus/types';
import NotFound from '@theme-original/NotFound';
import type NotFoundType from '@theme/NotFound';
import { Redirect } from 'react-router';

type Props = WrapperProps<typeof NotFoundType> & { location: Location };

export default function NotFoundWrapper(props: Props): ReactNode {
    const pathname = props?.location?.pathname;

    return <BrowserOnly>{() => <NotFound {...props} />}</BrowserOnly>;
}
