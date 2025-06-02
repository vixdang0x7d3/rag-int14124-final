#!/user/bin/env bash

if [ $# -eq 0 ]; then
		echo "Usage: $0 user@hostname [port] [remote_directory]"
		echo "Example: $0 john@server.com"
		echo "Example: $0 john@server.com 8080 ~/my-project"
		exit 1
fi

USER_HOST="$1"
PORT="${2:-2718}" 
REMOTE_DIR="${3:-$(pwd | sed 's|.*/||')}"  

echo "Starting marimo on $USER_HOST"
echo "Remote directory: $REMOTE_DIR"
echo "Local access: http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

# SSH with port forwarding and run marimo
ssh -L "$PORT:localhost:$PORT" "$USER_HOST" \
    "cd $REMOTE_DIR && uv run marimo edit --watch --host 0.0.0.0 --port $PORT"
