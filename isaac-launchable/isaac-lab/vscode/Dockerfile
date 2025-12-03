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
FROM nvcr.io/nvidia/isaac-lab:2.2.0

USER root
ENV OMNI_KIT_ALLOW_ROOT=1

RUN apt update && \
    apt install -y \
	build-essential \
	cmake \
	curl \
	wget \
	git

RUN curl -fsSL https://code-server.dev/install.sh | sh -s -- --version=4.96.4

RUN apt update && \
	apt install -y \
	unzip \
	dnsutils \
	iproute2 \
	jq

ENV WORKSPACE_DIR=/workspace

COPY --chown=ubuntu:ubuntu ./README.md ${WORKSPACE_DIR}/README.md
COPY --chown=ubuntu:ubuntu ./settings.json ${WORKSPACE_DIR}/.vscode/settings.json

RUN code-server \
    --install-extension ms-python.python \
    --install-extension ms-toolsai.jupyter \
    --force ${WORKSPACE_DIR}

COPY entrypoint.sh /entrypoint.sh

# OV Cache Directories
RUN mkdir -p /home/root/.cache/ov
RUN mkdir -p /home/root/.local/share/ov
RUN chown root:root /home/root/.cache/ov
RUN chown root:root /home/root/.local/share/ov

# Copy leisaac from local directory
COPY --chown=ubuntu:ubuntu ../../../leisaac ${WORKSPACE_DIR}/leisaac

# Copy assets from local leisaac-assets directory
COPY --chown=ubuntu:ubuntu ../../../leisaac-assets/so101_follower.usd ${WORKSPACE_DIR}/leisaac/assets/robots/so101_follower.usd
COPY --chown=ubuntu:ubuntu ../../../leisaac-assets/kitchen_with_orange.zip ${WORKSPACE_DIR}/leisaac/assets/scenes/kitchen_with_orange.zip
COPY --chown=ubuntu:ubuntu ../../../leisaac-assets/table_with_cube.zip ${WORKSPACE_DIR}/leisaac/assets/scenes/table_with_cube.zip
COPY --chown=ubuntu:ubuntu ../../../leisaac-assets/2025-10-27-k0QSo7z0mslightwheel_bedroom.zip ${WORKSPACE_DIR}/leisaac/assets/scenes/lightwheel_bedroom.zip
COPY --chown=ubuntu:ubuntu ../../../leisaac-assets/2025-07-18-BddImJA8hPlightwheel_toyroom.zip ${WORKSPACE_DIR}/leisaac/assets/scenes/lightwheel_toyroom.zip

RUN cd /workspace/leisaac/assets/scenes && \
	unzip kitchen_with_orange.zip && \
	unzip table_with_cube.zip && \
	unzip lightwheel_bedroom.zip && \
	unzip lightwheel_toyroom.zip && \
	cd .. && \
	cd .. && \
	cd ..

# Install leisaac package and dependencies
RUN cd ${WORKSPACE_DIR}/leisaac && \
	pip install -e source/leisaac && \
	pip install pyzmq
	
ENTRYPOINT ["/bin/bash"]

CMD ["/entrypoint.sh"]
