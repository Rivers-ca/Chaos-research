#!/bin/bash
#
# GitHub Activity Demo - Automated Setup Script
#
# This script automates the setup of the GitHub activity demonstration repository
# and configures macOS launchd for scheduled automatic runs.
#
# Usage: ./setup_demo.sh [options]
#   -r, --repo-path PATH      Path to create demo repository (default: ~/github-demo-repo)
#   -e, --email EMAIL         Git author email (required)
#   -n, --name NAME           Git author name (default: "Demo Bot")
#   -u, --remote-url URL      GitHub remote URL (optional, can be added later)
#   --ssh-only                Use SSH remote (requires key setup)
#   --no-launchd              Skip launchd installation
#   --hour-start HOUR         Start randomization hour (default: 9)
#   --hour-end HOUR           End randomization hour (default: 17)
#

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ${NC} $*"
}

log_success() {
    echo -e "${GREEN}✓${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $*"
}

log_error() {
    echo -e "${RED}✗${NC} $*"
}

# Defaults
REPO_PATH="${HOME}/github-demo-repo"
GIT_AUTHOR_EMAIL=""
GIT_AUTHOR_NAME="Demo Bot"
REMOTE_URL=""
USE_SSH=false
INSTALL_LAUNCHD=true
HOUR_START=9
HOUR_END=17

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--repo-path)
            REPO_PATH="$2"
            shift 2
            ;;
        -e|--email)
            GIT_AUTHOR_EMAIL="$2"
            shift 2
            ;;
        -n|--name)
            GIT_AUTHOR_NAME="$2"
            shift 2
            ;;
        -u|--remote-url)
            REMOTE_URL="$2"
            shift 2
            ;;
        --ssh-only)
            USE_SSH=true
            shift
            ;;
        --no-launchd)
            INSTALL_LAUNCHD=false
            shift
            ;;
        --hour-start)
            HOUR_START="$2"
            shift 2
            ;;
        --hour-end)
            HOUR_END="$2"
            shift 2
            ;;
        -h|--help)
            grep "^#" "$0" | grep -v "^#!/bin/bash" | sed 's/^# *//'
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$GIT_AUTHOR_EMAIL" ]; then
    log_error "Git author email is required"
    echo "Usage: $0 -e your.email@example.com [options]"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info "GitHub Activity Demo Setup"
log_info "Repository path: $REPO_PATH"
log_info "Author: $GIT_AUTHOR_NAME <$GIT_AUTHOR_EMAIL>"

# Create repository directory
if [ -d "$REPO_PATH" ]; then
    if [ -d "$REPO_PATH/.git" ]; then
        log_warn "Repository already exists at $REPO_PATH"
        read -p "Continue with existing repository? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Setup cancelled"
            exit 1
        fi
    else
        log_error "$REPO_PATH exists but is not a git repository"
        exit 1
    fi
else
    log_info "Creating repository directory: $REPO_PATH"
    mkdir -p "$REPO_PATH"
fi

# Initialize git repository if needed
cd "$REPO_PATH"

if [ ! -d .git ]; then
    log_info "Initializing git repository"
    git init
    git config user.name "$GIT_AUTHOR_NAME"
    git config user.email "$GIT_AUTHOR_EMAIL"

    # Create initial README
    cat > README.md << EOF
# GitHub Contribution Graph Demo

This repository is part of a school project demonstrating GitHub contribution graph visualization.

Generated commits are automatically created by \`demo_github_activity.py\`.

See \`DEMO_README.md\` for complete documentation.
EOF

    git add README.md
    git commit -m "initial: GitHub contribution graph demo"
    log_success "Repository initialized"
else
    log_success "Using existing repository"
    # Verify configuration
    if [ "$(git config user.email)" != "$GIT_AUTHOR_EMAIL" ]; then
        log_warn "Updating git author email to $GIT_AUTHOR_EMAIL"
        git config user.name "$GIT_AUTHOR_NAME"
        git config user.email "$GIT_AUTHOR_EMAIL"
    fi
fi

# Copy demo files
log_info "Copying demo files"

for file in demo_github_activity.py launch_demo.sh com.demo.github_activity.plist DEMO_README.md; do
    src="${SCRIPT_DIR}/${file}"
    if [ ! -f "$src" ]; then
        log_error "Missing source file: $src"
        exit 1
    fi

    if [ -f "$REPO_PATH/$file" ]; then
        log_warn "File exists, skipping: $file"
    else
        cp "$src" "$REPO_PATH/"
        log_success "Copied: $file"
    fi
done

# Make scripts executable
chmod +x "$REPO_PATH/demo_github_activity.py"
chmod +x "$REPO_PATH/launch_demo.sh"

# Create environment configuration
if [ ! -f "$REPO_PATH/.github_demo.env" ]; then
    log_info "Creating environment configuration"
    cat > "$REPO_PATH/.github_demo.env" << EOF
# GitHub Demo Configuration
export GIT_AUTHOR_NAME="$GIT_AUTHOR_NAME"
export GIT_AUTHOR_EMAIL="$GIT_AUTHOR_EMAIL"
EOF
    chmod 600 "$REPO_PATH/.github_demo.env"
    log_success "Created: .github_demo.env"
else
    log_warn ".github_demo.env already exists, skipping"
fi

# Add files to gitignore
log_info "Updating .gitignore"
{
    echo ".github_demo.env"
    echo "logs/"
    echo ".demo_commit_hashes"
} | while read -r pattern; do
    if ! grep -q "^${pattern}$" "$REPO_PATH/.gitignore" 2>/dev/null; then
        echo "$pattern" >> "$REPO_PATH/.gitignore"
    fi
done

if [ -z "$(git config user.email)" ] || ! git diff-index --quiet HEAD -- 2>/dev/null || [ ! -f .git/HEAD ]; then
    log_warn "Git not properly configured, skipping .gitignore commit"
else
    git add .gitignore
    git diff --cached --quiet || git commit -m "chore: add demo configuration to gitignore"
fi

log_success "Updated: .gitignore"

# Create directories
mkdir -p "$REPO_PATH/logs"
mkdir -p "$REPO_PATH/sensor_data"
log_success "Created directories: logs/, sensor_data/"

# Handle remote repository
if [ -n "$REMOTE_URL" ]; then
    log_info "Adding remote repository"

    # Check if remote already exists
    if git remote get-url origin &>/dev/null; then
        EXISTING_REMOTE=$(git remote get-url origin)
        if [ "$EXISTING_REMOTE" != "$REMOTE_URL" ]; then
            log_warn "Remote already configured: $EXISTING_REMOTE"
            read -p "Update to $REMOTE_URL? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                git remote set-url origin "$REMOTE_URL"
                log_success "Updated remote URL"
            fi
        fi
    else
        git remote add origin "$REMOTE_URL"
        log_success "Added remote: $REMOTE_URL"
    fi
else
    log_warn "No remote URL provided. Add it later with:"
    echo "  cd $REPO_PATH"
    echo "  git remote add origin <url>"
fi

# Configure authentication
log_info "Checking Git authentication"
if [ "$USE_SSH" = true ] || [[ "${REMOTE_URL:-}" == git@github.com:* ]]; then
    log_info "Using SSH authentication"
    if ! ssh -T git@github.com &>/dev/null; then
        log_warn "SSH key not configured or not loaded"
        echo "To set up SSH:"
        echo "  1. ssh-keygen -t ed25519 -C '$GIT_AUTHOR_EMAIL'"
        echo "  2. Add the key to GitHub: https://github.com/settings/keys"
        echo "  3. Test with: ssh -T git@github.com"
    else
        log_success "SSH authentication working"
    fi
else
    log_warn "Using HTTPS (SSH recommended). To use SSH:"
    echo "  1. Set up SSH key as shown above"
    echo "  2. Update remote: git remote set-url origin git@github.com:username/repo.git"
fi

# Install launchd job
if [ "$INSTALL_LAUNCHD" = true ]; then
    log_info "Setting up launchd scheduler"

    PLIST_PATH="${HOME}/Library/LaunchAgents/com.demo.github_activity.plist"
    mkdir -p "$(dirname "$PLIST_PATH")"

    # Update launch script with correct paths
    sed \
        -e "s|^\(REPO_PATH=\).*|\1\"$REPO_PATH\"|" \
        -e "s|^\(HOUR_START=\).*|\1$HOUR_START|" \
        -e "s|^\(HOUR_END=\).*|\1$HOUR_END|" \
        "$REPO_PATH/launch_demo.sh" > "$REPO_PATH/launch_demo_configured.sh"
    chmod +x "$REPO_PATH/launch_demo_configured.sh"

    # Create plist
    sed \
        -e "s|__REPO_PATH__|$REPO_PATH|g" \
        -e "s|__LAUNCH_SCRIPT_PATH__|$REPO_PATH/launch_demo_configured.sh|g" \
        "$REPO_PATH/com.demo.github_activity.plist" > "$PLIST_PATH"

    chmod 644 "$PLIST_PATH"
    log_success "Created launchd plist: $PLIST_PATH"

    # Load launchd job
    if launchctl load "$PLIST_PATH" 2>/dev/null; then
        log_success "Loaded launchd job"
        echo "To verify: launchctl list | grep github_activity"
    else
        log_warn "Could not load launchd job (might need to unload old version first)"
        echo "Try: launchctl unload $PLIST_PATH && launchctl load $PLIST_PATH"
    fi
else
    log_info "Skipping launchd installation"
fi

# Test the script
log_info "Testing script"
echo ""
if python3 "$REPO_PATH/demo_github_activity.py" \
    --check-only 2>&1 | grep -q "valid"; then
    log_success "Configuration is valid"
else
    log_warn "Configuration test failed. Run manually to diagnose:"
    echo "  export GITHUB_DEMO_REPO='$REPO_PATH'"
    echo "  export GIT_AUTHOR_EMAIL='$GIT_AUTHOR_EMAIL'"
    echo "  python3 '$REPO_PATH/demo_github_activity.py' --verbose"
fi

# Summary
log_success "Setup complete!"
echo ""
log_info "Next steps:"
echo "1. Set up authentication:"
echo "   - SSH: https://github.com/settings/keys"
echo "   - Token: Set GIT_AUTHOR_EMAIL in $REPO_PATH/.github_demo.env"
echo ""
echo "2. Add remote repository (if not done):"
echo "   cd $REPO_PATH"
echo "   git remote add origin <github-url>"
echo "   git push -u origin main"
echo ""
echo "3. Test the script:"
echo "   cd $REPO_PATH"
echo "   export GITHUB_DEMO_REPO='$REPO_PATH'"
echo "   export GIT_AUTHOR_EMAIL='$GIT_AUTHOR_EMAIL'"
echo "   python3 demo_github_activity.py --dry-run"
echo ""
echo "4. Monitor execution:"
echo "   tail -f $REPO_PATH/logs/demo.log"
echo ""
echo "For complete documentation, see:"
echo "  $REPO_PATH/DEMO_README.md"
