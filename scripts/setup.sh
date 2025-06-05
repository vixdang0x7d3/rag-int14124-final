#!/usr/bin/env bash

curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
apt update && apt install gh

gh config set git_protocol https
gh auth login --web
gh auth setup-git

git config --global user.name "vixdang0x7d3"
git config --global user.email "buttfacecat2211@gmail.com"

curl -LsSf https://astral.sh/uv/install.sh | sh

if [ -f "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
elif [ -f "/root/.local/bin/uv" ]; then
    export PATH="/root/.local/bin:$PATH"
else
    echo "uv not found in expected locations"
    find / -name "uv" -type f 2>/dev/null | head -5
    exit 1
fi

echo "checking uv installation..."
uv --version

echo "installing dependencies..."
uv sync --frozen

read -p "start marimo notebook server? (y/n): " answer

if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "starting marimo notebook"
    uv run marimo edit --headless --no-token --port 1337
else
    echo "marimo server lauch skipped"
fi
