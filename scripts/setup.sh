#!/usr/bin/env bash

curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
apt update && apt install gh

gh config set git_protocol https
gh auth login --web
gh auth setup-git

curl -LsSf https://astral.sh/uv/install.sh | sh

if [ -f "$HOME/.cargo/bin/uv" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
elif [ -f "/root/.cargo/bin/uv" ]; then
    export PATH="/root/.cargo/bin:$PATH"
else
    echo "uv not found in expected locations"
    find / -name "uv" -type f 2>/dev/null | head -5
    exit 1
fi

echo "checking uv installation..."
uv --version

echo "installing dependencies"
uv sync --frozen

echo "starting marimo notebook"
uv sync --frozen
