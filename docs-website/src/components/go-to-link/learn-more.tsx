import { FC } from 'react';

import { GoToLink } from '@site/src/components/go-to-link/go-to-link';

type GoToDocumentationProps = {
    link: string;
};

export const LearnMore: FC<GoToDocumentationProps> = ({ link }) => {
    return <GoToLink link={link} name={'Learn more'} />;
};
