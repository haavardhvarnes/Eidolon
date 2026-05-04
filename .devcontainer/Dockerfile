FROM node:20

ARG TZ
ARG JULIA_VERSION=1.11
ENV TZ="$TZ"

RUN apt-get update && apt-get install -y --no-install-recommends \
        less git procps sudo fzf zsh man-db unzip gnupg2 \
        gh iptables ipset iproute2 dnsutils aggregate jq curl ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Persistent zsh history across rebuilds (devcontainer mounts /commandhistory)
ARG USERNAME=node
RUN mkdir -p /commandhistory \
    && touch /commandhistory/.zsh_history \
    && chown -R $USERNAME /commandhistory

ENV DEVCONTAINER=true

# Workspace + Claude state dirs (volume-mounted; ensure they exist)
RUN mkdir -p /workspace /home/node/.claude /home/node/.julia \
    && chown -R node:node /workspace /home/node/.claude /home/node/.julia

WORKDIR /workspace

# Allow npm global installs without root
RUN ARCH=$(dpkg --print-architecture) \
    && wget -qO- "https://github.com/dandavison/delta/releases/download/0.18.2/git-delta_0.18.2_${ARCH}.deb" -o /tmp/delta.deb \
    && (dpkg -i /tmp/delta.deb || true) \
    && rm -f /tmp/delta.deb

USER node

# zsh + powerlevel10k for a sane prompt
RUN sh -c "$(curl -fsLS https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone --depth=1 https://github.com/romkatv/powerlevel10k.git \
        ${ZSH_CUSTOM:-/home/node/.oh-my-zsh/custom}/themes/powerlevel10k \
    && sed -i 's|ZSH_THEME=.*|ZSH_THEME="powerlevel10k/powerlevel10k"|' /home/node/.zshrc \
    && echo 'HISTFILE=/commandhistory/.zsh_history' >> /home/node/.zshrc \
    && echo 'export PATH="/home/node/.npm-global/bin:/usr/local/julia/bin:$PATH"' >> /home/node/.zshrc

# Per-user npm global so we don't need root for `npm i -g claude-code`
RUN mkdir -p /home/node/.npm-global \
    && npm config set prefix /home/node/.npm-global \
    && npm install -g @anthropic-ai/claude-code

# Julia via the official tarball (juliaup needs network at run time)
USER root
RUN ARCH=$(uname -m | sed 's/x86_64/x64/;s/aarch64/aarch64/') \
    && curl -fsSL "https://julialang-s3.julialang.org/bin/linux/${ARCH%-*}/${JULIA_VERSION}/julia-${JULIA_VERSION}-latest-linux-${ARCH}.tar.gz" -o /tmp/julia.tgz \
    && mkdir -p /usr/local/julia \
    && tar -xzf /tmp/julia.tgz -C /usr/local/julia --strip-components=1 \
    && rm /tmp/julia.tgz \
    && ln -sf /usr/local/julia/bin/julia /usr/local/bin/julia

# Firewall init script — runs once via postCreateCommand under sudo
COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh \
    && echo "node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" \
        > /etc/sudoers.d/node-firewall \
    && chmod 0440 /etc/sudoers.d/node-firewall

USER node
ENV SHELL=/bin/zsh
CMD ["zsh"]
