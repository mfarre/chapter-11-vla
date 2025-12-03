/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { AppStreamer, StreamEvent, StreamProps, DirectConfig, GFNConfig } from '@nvidia/omniverse-webrtc-streaming-library';
import { GetConfig } from './config';


interface AppStreamProps {
    sessionId: string
    backendUrl: string
    signalingserver: string
    signalingport: number
    mediaserver: string
    mediaport: number
    accessToken: string
    style?: React.CSSProperties;
    onStarted: () => void;
    onStreamFailed: () => void;
    onLoggedIn: (userId: string) => void;
    handleCustomEvent: (event: any) => void;
    onFocus: () => void;
    onBlur: () => void;
}

interface AppStreamState {
    streamReady: boolean;
}

export default class AppStream extends Component<AppStreamProps, AppStreamState> {
    private _requested: boolean;

    static defaultProps = {
        style: {}
    };

    static propTypes = {
        onStarted: PropTypes.func.isRequired,
        handleCustomEvent: PropTypes.func.isRequired,
        style: PropTypes.object
    };

    constructor(props: AppStreamProps) {
        super(props);

        this._requested = false;
        this.state = {
            streamReady: false
        };
    }

    componentDidMount() {
        if (!this._requested) {
            this._requested = true;

            let streamProps: StreamProps;
            let streamConfig: DirectConfig | GFNConfig;
            let streamSource: 'gfn' | 'direct';

            if (GetConfig().source === 'gfn') {
                    streamSource = 'gfn'
                    streamConfig = {
                        //@ts-ignore
                        GFN             : GFN,
                        catalogClientId : GetConfig().gfn.catalogClientId,
                        clientId        : GetConfig().gfn.clientId,
                        cmsId           : GetConfig().gfn.cmsId,
                        onUpdate        : (message: StreamEvent) => this._onUpdate(message),
                        onStart         : (message: StreamEvent) => this._onStart(message),
                        onCustomEvent   : (message: any) => this._onCustomEvent(message)
                    }
            }

            else if (GetConfig().source === 'local') {
                streamSource = 'direct';
                streamConfig = {
                    videoElementId: 'remote-video',
                    audioElementId: 'remote-audio',
                    authenticate: false,
                    maxReconnects: 20,
                    server: "",
                    signalingServer: GetConfig().local.server,
                    mediaServer: GetConfig().local.server,
                    signalingPort: 49100,
                    mediaPort: 1024,
                    nativeTouchEvents: true,
                    width: 1920,
                    height: 1080,
                    fps: 60,
                    onUpdate: (message: StreamEvent) => this._onUpdate(message),
                    onStart: (message: StreamEvent) => this._onStart(message),
                    onCustomEvent: (message: any) => this._onCustomEvent(message),
                    onStop: (message: StreamEvent) => { console.log(message) },
                    onTerminate: (message: StreamEvent) => { console.log(message) }
                };
            }
                
            else if (GetConfig().source === 'stream') {
                streamSource = 'direct'
                streamConfig = {
                    signalingServer: this.props.signalingserver,
                    signalingPort: this.props.signalingport,
                    mediaServer: this.props.mediaserver,
                    mediaPort: this.props.mediaport,
                    backendUrl: this.props.backendUrl,
                    sessionId: this.props.sessionId,
                    autoLaunch: true,
                    cursor: 'free',
                    mic: false,
                    videoElementId: 'remote-video',
                    audioElementId: 'remote-audio',
                    authenticate: false,
                    maxReconnects: 20,
                    nativeTouchEvents: true,
                    width: 1920,
                    height: 1080,
                    fps: 60,
                    onUpdate: (message: StreamEvent) => this._onUpdate(message),
                    onStart: (message: StreamEvent) => this._onStart(message),
                    onCustomEvent: (message: any) => this._onCustomEvent(message),
                    onStop: (message: StreamEvent) => { console.log(message) },
                    onTerminate: (message: StreamEvent) => { console.log(message) },
                };
            }
                
            else {
                console.error(`Unknown stream source: ${GetConfig().source}`);
                return
            }

            try {
                streamProps = {streamConfig, streamSource}
                AppStreamer.connect(streamProps)
                .then((result: StreamEvent) => {
                    console.info(result);
                })
                .catch((error: StreamEvent) => {
                    console.error(error);
                });
            }
            catch (error) {
                console.error(error);
            }
        }
    }

    componentDidUpdate(_prevProps: AppStreamProps, prevState: AppStreamState, _snapshot: any) {
        if (prevState.streamReady === false && this.state.streamReady === true) {
            const player = document.getElementById("gfn-stream-player-video") as HTMLVideoElement;
            
            if (player) {
                if (GetConfig().source === "gfn")
                {
                    player.style.position = "relative";
                    const container = document.getElementById("gfn-stream-player-video-container") as HTMLVideoElement;
                    container.style.background = "white";
                }

                player.tabIndex = -1;
                player.playsInline = true;
                player.muted = true;
                player.play();
            }
        }
    }

    static sendMessage(message: any) {
        AppStreamer.sendMessage(message);
    }

    static stop() {
        AppStreamer.stop();
        (AppStreamer as any)._stream = null; // Accessing a private member
    }

    _onStart(message: any) {
        if (message.action === 'start' && message.status === 'success' && !this.state.streamReady) {
            console.info('streamReady');
            this.setState({ streamReady: true });
            this.props.onStarted();
        }

        if (message.status === "error" && GetConfig().source === "stream")
        {
            console.log(message.info);
            alert(message.info);
            this.props.onStreamFailed();
            return;
        }
    }

    _onUpdate(message: any) {
        try {
            if (message.action === 'authUser' && message.status === 'success') {
                this.props.onLoggedIn(message.info);
            }
        } catch (error) {
            console.error(message);
        }
    }

    _onCustomEvent(message: any) {
        this.props.handleCustomEvent(message);
    }

    _onStop(message: any) {
        console.info('Stream stopped', message);
    }

    _onTerminate(message: any) {
        console.info('Stream terminated', message);
    }

    render() {
        const source = GetConfig().source;

        if (source === 'gfn') {
            return (
                <div
                    id="view"
                    style={{
                        backgroundColor: this.state.streamReady ? 'white': '#dddddd',
                        display: 'flex', justifyContent: 'space-between',
                        height: "100%",
                        width: "100%",
                        ...this.props.style
                    }}
                />
            );
        } else if (source === 'local' || source === 'stream') {
            return (
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    width: '100%',
                    height: '100%'
                }}>
                    <video
                        key={'video-canvas'}
                        id={'remote-video'}
                        style={{
                            maxWidth: '100vw',
                            maxHeight: '100vh',
                            objectFit: 'cover'
                        }}
                        tabIndex={-1}
                        playsInline muted
                        autoPlay
                    />
                    <audio id="remote-audio" muted></audio>
                    <h3 style={{ visibility: 'hidden' }} id="message-display">...</h3>
                </div>
            );
        }

        return null;
    }
}
