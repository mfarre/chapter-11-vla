#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -xeuo pipefail

ENV=${ENV:-instance}

set_config() {
  cp ./stream.config.json ./stream.config.json.tmp
  jq \
    "${@}" \
    ./stream.config.json.tmp > ./stream.config.json
}

main() {
  case ${ENV} in
    instance)
      IP=$(nslookup $(hostname) | grep Address | head -2 | tail -1 | awk '{print $2}')
      ;;
    localhost)
      IP="127.0.0.1"
      ;;
    brev)
      IP=$(curl https://icanhazip.com)
      ;;
    *)
      echo "Env ${ENV} not understood"
      exit 1
      ;;
  esac

  set_config '.source = "local"'
  set_config ".local.server = \"${IP}\""

  exec npm run dev
}

main "${@}"
