import { ComponentProps, CSSProperties, FC, ReactNode } from 'react';

import Heading from '@theme/Heading';
import clsx from 'clsx';

import { AutoPlayVideo } from '../auto-play-video/auto-play-video';

import styles from './styles.module.css';

const SectionDivider = () => {
    return <hr className={styles.divider} />;
};

type SectionColumnProps = {
    children: ReactNode;
    center?: boolean;
    flex?: CSSProperties['flex'];
    className?: string;
    videoColumn?: boolean;
};

const SectionColumn: FC<SectionColumnProps> = ({ children, center, flex = 1, videoColumn = false, className }) => {
    return (
        <div
            className={clsx(
                videoColumn ? styles.videoColumn : {},
                styles.column,
                { [styles.columnCenter]: center },
                className
            )}
            style={{ flex }}
        >
            {children}
        </div>
    );
};

type SectionRowProps = {
    children: ReactNode;
    gap?: CSSProperties['gap'];
    alignItems?: CSSProperties['alignItems'];
    justifyContent?: CSSProperties['justifyContent'];
    className?: string;
};

const SectionRow: FC<SectionRowProps> = ({ children, gap = '5rem', alignItems, justifyContent, className }) => {
    return (
        <div className={clsx(styles.sectionRow, className)} style={{ gap, alignItems, justifyContent }}>
            {children}
        </div>
    );
};

type SectionListTitleProps = {
    children: ReactNode;
};

const SectionListTitle: FC<SectionListTitleProps> = ({ children }) => {
    return (
        <Heading as={'h4'} className={styles.listTitle}>
            {children}
        </Heading>
    );
};

type SectionListDescriptionProps = {
    children: ReactNode;
};

const SectionListDescription: FC<SectionListDescriptionProps> = ({ children }) => {
    return <p className={styles.listDescription}>{children}</p>;
};

type SectionListProps = {
    children: ReactNode;
};

const SectionList: FC<SectionListProps> = ({ children }) => {
    return <ul className={styles.sectionList}>{children}</ul>;
};

type SectionListItemProps = {
    children: ReactNode;
};

const SectionListItem: FC<SectionListItemProps> = ({ children }) => {
    return <li className={styles.listItem}>{children}</li>;
};

type SectionListContainerProps = {
    children: ReactNode;
};

const SectionListContainer: FC<SectionListContainerProps> = ({ children }) => {
    return <div>{children}</div>;
};

type SectionDescriptionProps = {
    children: ReactNode;
    marginTop?: CSSProperties['marginTop'];
    marginBottom?: CSSProperties['marginBottom'];
    center?: boolean;
};

const SectionDescription: FC<SectionDescriptionProps> = ({ children, marginBottom, marginTop, center }) => {
    return (
        <p className={clsx(styles.description, { [styles.center]: center })} style={{ marginBottom, marginTop }}>
            {children}
        </p>
    );
};

type SectionVideoProps = {
    videoUrl: string;
};

const SectionVideo: FC<SectionVideoProps> = ({ videoUrl }) => {
    return (
        <div className={styles.videoSection}>
            <AutoPlayVideo videoUrl={videoUrl} />
        </div>
    );
};

type SectionTitleProps = {
    children: ReactNode;
    as?: ComponentProps<typeof Heading>['as'];
    center?: boolean;
};

const SectionTitle: FC<SectionTitleProps> = ({ children, as = 'h1', center }) => {
    return (
        <Heading as={as} className={clsx(styles.title, { [styles.center]: center })}>
            {children}
        </Heading>
    );
};

type SectionProps = {
    children: ReactNode;
    withBackground?: boolean;
    withPadding?: boolean;
    style?: CSSProperties;
};

export const Section = ({ children, withBackground = false, withPadding, style }: SectionProps) => {
    return (
        <section
            className={clsx({
                [styles.sectionBackground]: withBackground,
                [styles.sectionPadding]: withPadding,
            })}
            style={style}
        >
            {children}
        </section>
    );
};

Section.Title = SectionTitle;
Section.Description = SectionDescription;
Section.Video = SectionVideo;
Section.ListContainer = SectionListContainer;
Section.List = SectionList;
Section.ListItem = SectionListItem;
Section.ListDescription = SectionListDescription;
Section.ListTitle = SectionListTitle;
Section.Column = SectionColumn;
Section.Divider = SectionDivider;
Section.Row = SectionRow;
