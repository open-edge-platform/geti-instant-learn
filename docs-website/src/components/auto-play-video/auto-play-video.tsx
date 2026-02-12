import { RefObject, useEffect, useLayoutEffect, useRef, useState } from 'react';

const useAutoPlayVideo = (isPlaying: boolean) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const isInView = useRef(false);
    const observer = useRef<IntersectionObserver>();

    useLayoutEffect(() => {
        observer.current = new IntersectionObserver(
            ([entry]) => {
                if (videoRef.current) {
                    isInView.current = entry.isIntersecting;

                    if (entry.isIntersecting && isPlaying) {
                        videoRef.current.play();
                    } else {
                        videoRef.current.pause();
                    }
                }
            },
            { threshold: 0.5 }
        );
    }, []);

    useEffect(() => {
        if (videoRef.current) {
            observer.current.observe(videoRef.current);
        }

        return () => {
            if (videoRef.current) {
                observer.current.unobserve(videoRef.current);
            }
        };
    }, []);

    // Auto play video if it is in a video group
    useEffect(() => {
        if (videoRef.current && isPlaying && isInView.current) {
            videoRef.current.play();
        } else if (videoRef.current) {
            videoRef.current.pause();
        }
    }, [isPlaying]);

    return videoRef;
};

interface AutoPlayVideoProps {
    videoUrl: string;
    onEnded?: () => void;
    isPlaying?: boolean;
    controls?: boolean;
}

const ActualAutoPlayVideo = ({
    videoUrl,
    onEnded,
    isPlaying = true,
    controls = true,
}: {
    videoUrl: string;
    onEnded?: () => void;
    isPlaying?: boolean;
    controls?: boolean;
}) => {
    const videoRef = useAutoPlayVideo(isPlaying);

    return (
        <video
            ref={videoRef}
            controls={controls}
            width={'100%'}
            preload='metadata'
            loop={onEnded === undefined}
            muted
            onEnded={onEnded}
        >
            <source src={videoUrl} type='video/mp4' />
        </video>
    );
};

// Add a small delay before showing videos to prevent a slow page load
const DELAY_SHOWING_VIDEOS = 100;

export const AutoPlayVideo = (props: AutoPlayVideoProps) => {
    const [isVisible, setIsVisible] = useState(false);
    const timer = useRef<ReturnType<typeof setTimeout | null>>(null);

    useEffect(() => {
        timer.current = setTimeout(() => setIsVisible(true), DELAY_SHOWING_VIDEOS);

        return () => {
            if (timer.current) {
                clearTimeout(timer.current);
            }
        };
    }, []);

    return isVisible ? <ActualAutoPlayVideo {...props} /> : null;
};
