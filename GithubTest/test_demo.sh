#!/bin/bash
#
# GitHub Activity Demo - Testing & Verification Script
#
# Validates the demo setup and performs safe tests before enabling automation.
#
# Usage: ./test_demo.sh [options]
#   -r, --repo-path PATH      Path to demo repository (default: ~/github-demo-repo)
#   -q, --quick               Quick validation only (no actual commits)
#   -f, --full                Full test including actual commits (will push)
#

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

test_pass() {
    echo -e "${GREEN}✓${NC} $*"
}

test_fail() {
    echo -e "${RED}✗${NC} $*"
}

test_warn() {
    echo -e "${YELLOW}⚠${NC} $*"
}

test_info() {
    echo -e "${BLUE}ℹ${NC} $*"
}

# Defaults
REPO_PATH="${HOME}/github-demo-repo"
TEST_MODE="quick"  # quick, full

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--repo-path)
            REPO_PATH="$2"
            shift 2
            ;;
        -q|--quick)
            TEST_MODE="quick"
            shift
            ;;
        -f|--full)
            TEST_MODE="full"
            shift
            ;;
        -h|--help)
            grep "^#" "$0" | grep -v "^#!/bin/bash" | sed 's/^# *//'
            exit 0
            ;;
        *)
            test_fail "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$REPO_PATH" || {
    test_fail "Repository path does not exist: $REPO_PATH"
    exit 1
}

test_info "GitHub Activity Demo - Test Suite"
test_info "Repository: $REPO_PATH"
test_info "Test mode: $TEST_MODE"
echo ""

TESTS_PASSED=0
TESTS_FAILED=0

# Test 1: Repository structure
test_info "Test 1: Repository structure"
{
    [ -d .git ] && test_pass ".git directory exists" && ((TESTS_PASSED++)) || {
        test_fail ".git directory not found"
        ((TESTS_FAILED++))
    }

    [ -f demo_github_activity.py ] && test_pass "demo_github_activity.py exists" && ((TESTS_PASSED++)) || {
        test_fail "demo_github_activity.py not found"
        ((TESTS_FAILED++))
    }

    [ -f launch_demo.sh ] && test_pass "launch_demo.sh exists" && ((TESTS_PASSED++)) || {
        test_fail "launch_demo.sh not found"
        ((TESTS_FAILED++))
    }

    [ -f .github_demo.env ] && test_pass ".github_demo.env exists" && ((TESTS_PASSED++)) || {
        test_warn ".github_demo.env not found (will use environment variables)"
    }

    [ -d sensor_data ] && test_pass "sensor_data directory exists" && ((TESTS_PASSED++)) || {
        test_warn "sensor_data directory not found (will be created)"
    }

    [ -d logs ] && test_pass "logs directory exists" && ((TESTS_PASSED++)) || {
        test_warn "logs directory not found (will be created)"
    }
}
echo ""

# Test 2: Git configuration
test_info "Test 2: Git configuration"
{
    GIT_NAME=$(git config user.name 2>/dev/null || echo "")
    GIT_EMAIL=$(git config user.email 2>/dev/null || echo "")

    [ -n "$GIT_NAME" ] && test_pass "Git author name: $GIT_NAME" && ((TESTS_PASSED++)) || {
        test_fail "Git author name not set"
        ((TESTS_FAILED++))
    }

    [ -n "$GIT_EMAIL" ] && test_pass "Git author email: $GIT_EMAIL" && ((TESTS_PASSED++)) || {
        test_fail "Git author email not set"
        ((TESTS_FAILED++))
    }

    [ -n "$GIT_EMAIL" ] && export GIT_AUTHOR_EMAIL="$GIT_EMAIL"
    [ -n "$GIT_NAME" ] && export GIT_AUTHOR_NAME="$GIT_NAME"
}
echo ""

# Test 3: Environment variables
test_info "Test 3: Environment variables"
{
    if [ -z "${GIT_AUTHOR_EMAIL:-}" ]; then
        test_fail "GIT_AUTHOR_EMAIL not set"
        ((TESTS_FAILED++))
    else
        test_pass "GIT_AUTHOR_EMAIL: $GIT_AUTHOR_EMAIL"
        ((TESTS_PASSED++))
    fi

    if [ -z "${GIT_AUTHOR_NAME:-}" ]; then
        test_warn "GIT_AUTHOR_NAME not set (will default to 'Demo Bot')"
    else
        test_pass "GIT_AUTHOR_NAME: $GIT_AUTHOR_NAME"
        ((TESTS_PASSED++))
    fi
}
echo ""

# Test 4: Python script validation
test_info "Test 4: Python script validation"
{
    if python3 demo_github_activity.py --check-only 2>&1 | grep -q "valid"; then
        test_pass "Script configuration is valid"
        ((TESTS_PASSED++))
    else
        test_fail "Script validation failed"
        test_info "Run for details: python3 demo_github_activity.py --check-only --verbose"
        ((TESTS_FAILED++))
    fi
}
echo ""

# Test 5: Python dependencies
test_info "Test 5: Python dependencies"
{
    if python3 -c "import subprocess, json, logging, pathlib, hashlib, datetime, argparse, random" 2>/dev/null; then
        test_pass "All required Python modules available"
        ((TESTS_PASSED++))
    else
        test_fail "Missing Python modules"
        ((TESTS_FAILED++))
    fi
}
echo ""

# Test 6: Git remote
test_info "Test 6: Git remote"
{
    if git remote get-url origin &>/dev/null; then
        REMOTE=$(git remote get-url origin)
        test_pass "Remote configured: $REMOTE"
        ((TESTS_PASSED++))

        # Test connectivity
        if [[ "$REMOTE" == git@github.com:* ]]; then
            if ssh -T git@github.com &>/dev/null; then
                test_pass "SSH connection successful"
                ((TESTS_PASSED++))
            else
                test_warn "SSH connection failed (check key is loaded: ssh-add -l)"
                ((TESTS_FAILED++))
            fi
        fi
    else
        test_warn "No remote configured yet"
        test_info "Add with: git remote add origin <url>"
    fi
}
echo ""

# Test 7: Git operations
test_info "Test 7: Git operations"
{
    if git status &>/dev/null; then
        test_pass "Git repository is accessible"
        ((TESTS_PASSED++))
    else
        test_fail "Cannot access git repository"
        ((TESTS_FAILED++))
    fi

    # Check for uncommitted changes
    if [ -z "$(git status -s)" ]; then
        test_pass "Working directory clean"
        ((TESTS_PASSED++))
    else
        test_warn "Uncommitted changes present:"
        git status -s
    fi
}
echo ""

# Test 8: Dry-run mode
test_info "Test 8: Dry-run mode"
{
    test_info "Running script in dry-run mode..."
    if python3 demo_github_activity.py --dry-run 2>&1 | tee /tmp/demo_dryrun.log | grep -q "DRY RUN"; then
        test_pass "Dry-run mode works"
        ((TESTS_PASSED++))

        # Check that no actual files were created
        if [ ! -d sensor_data ] || [ -z "$(find sensor_data -type f 2>/dev/null)" ]; then
            test_pass "No actual data files created in dry-run"
            ((TESTS_PASSED++))
        else
            test_warn "Unexpected files created in dry-run mode"
        fi
    else
        test_fail "Dry-run mode failed"
        ((TESTS_FAILED++))
    fi
}
echo ""

# Test 9: Full test (optional)
if [ "$TEST_MODE" = "full" ]; then
    test_info "Test 9: Full execution test (creating actual commits)"
    echo -e "${YELLOW}Warning: This will create real commits.${NC}"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_info "Running full test..."
        if python3 demo_github_activity.py --verbose 2>&1 | tee -a /tmp/demo_test.log; then
            test_pass "Script executed successfully"
            ((TESTS_PASSED++))

            # Verify commits were created
            COMMIT_COUNT=$(git log --oneline -5 | grep "data:" | wc -l)
            if [ "$COMMIT_COUNT" -gt 0 ]; then
                test_pass "Created $COMMIT_COUNT commits"
                ((TESTS_PASSED++))
                test_info "Recent commits:"
                git log --oneline -3
            else
                test_warn "No commits found (may have been filtered as duplicates)"
            fi

            # Check sensor_data directory
            if [ -d sensor_data ] && [ -n "$(find sensor_data -type f -name '*.json' 2>/dev/null)" ]; then
                test_pass "Data files created in sensor_data/"
                ((TESTS_PASSED++))
                test_info "Sample data file:"
                find sensor_data -type f -name '*.json' | head -1 | xargs head -10
            else
                test_fail "No data files found"
                ((TESTS_FAILED++))
            fi

            # Try pushing if remote is configured
            if git remote get-url origin &>/dev/null; then
                test_info "Attempting to push commits..."
                if git push 2>&1; then
                    test_pass "Push successful"
                    ((TESTS_PASSED++))
                else
                    test_warn "Push failed (check authentication)"
                    test_info "To retry: cd $REPO_PATH && git push"
                    ((TESTS_FAILED++))
                fi
            fi
        else
            test_fail "Script execution failed"
            ((TESTS_FAILED++))
        fi
    else
        test_info "Skipping full test"
    fi
fi
echo ""

# Test 10: launchd configuration (if installed)
test_info "Test 10: launchd configuration"
{
    PLIST_PATH="${HOME}/Library/LaunchAgents/com.demo.github_activity.plist"
    if [ -f "$PLIST_PATH" ]; then
        test_pass "launchd plist exists: $PLIST_PATH"
        ((TESTS_PASSED++))

        if launchctl list | grep -q github_activity; then
            test_pass "launchd job is loaded"
            ((TESTS_PASSED++))
        else
            test_warn "launchd job not loaded"
            test_info "Load with: launchctl load $PLIST_PATH"
        fi
    else
        test_warn "launchd plist not installed (run setup_demo.sh to install)"
    fi
}
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_info "Test Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_pass "Passed: $TESTS_PASSED"
test_fail "Failed: $TESTS_FAILED"

if [ "$TESTS_FAILED" -eq 0 ]; then
    echo ""
    test_pass "All tests passed! Repository is ready."
    echo ""
    echo "Next steps:"
    echo "1. Verify dry-run output above"
    echo "2. Run full test: ./test_demo.sh --full"
    echo "3. Monitor execution: tail -f logs/demo.log"
    exit 0
else
    echo ""
    test_fail "Some tests failed. Please fix the issues above."
    echo ""
    echo "Common issues:"
    echo "- Set GIT_AUTHOR_EMAIL: export GIT_AUTHOR_EMAIL='your.email@example.com'"
    echo "- Set up Git remote: git remote add origin <url>"
    echo "- Configure SSH: ssh-keygen -t ed25519"
    echo ""
    echo "For detailed help, see: DEMO_README.md"
    exit 1
fi
