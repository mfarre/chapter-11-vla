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

import StreamConfig from "../stream.config.json";

type source = "gfn" | "local" | "stream";

type Config = {
  source: source;
  stream: {
    appServer: string;
    streamServer: string;
  };
  gfn: {
    catalogClientId: string;
    clientId: string;
    cmsId: number;
  };
  local: {
    server: string;
  };
};

function loadWindowVar<T>(v: string): T | null {
  let w = window as any;
  if (typeof w !== "undefined" && v in w) {
    return w[v] as T;
  } else {
    return null;
  }
}

function GetConfig(): Config {
  let source = loadWindowVar<string>("OVAS_SOURCE");
  if (source !== null) {
    StreamConfig.source = source!;
  }

  let localServer = loadWindowVar<string>("OVAS_LOCAL_SERVER");
  if (localServer !== null) {
    StreamConfig.local.server = localServer!;
  }

  let appURL = loadWindowVar<string>("OVAS_STREAM_APPSERVER");
  if (appURL !== null) {
    StreamConfig.stream.appServer = appURL!;
  }

  let streamURL = loadWindowVar<string>("OVAS_STREAM_STREAMSERVER");
  if (streamURL !== null) {
    StreamConfig.stream.streamServer = streamURL!;
  }

  return StreamConfig as Config;
}

export { GetConfig };
