#!/usr/bin/env python3
"""
GitHub Contribution Graph Demo: Autonomous Data Generation

This script generates simulated sensor data and commits it to a Git repository
to demonstrate how GitHub contribution graphs visualize changing daily activity.

Configuration is read from environment variables and the current repository state.
All operations are logged and can be previewed with --dry-run.
"""

import os
import sys
import json
import random
import logging
import subprocess
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('github_activity_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GitConfig:
    """Manages Git configuration from environment variables."""

    def __init__(self):
        self.repo_path = Path(os.getenv('GITHUB_DEMO_REPO', '.')).resolve()
        self.author_name = os.getenv('GIT_AUTHOR_NAME', 'Demo Bot')
        self.author_email = os.getenv('GIT_AUTHOR_EMAIL')
        self.dry_run = False

        if not self.author_email:
            raise ValueError(
                'GIT_AUTHOR_EMAIL not set. Set it with:\n'
                '  export GIT_AUTHOR_EMAIL="your.email@example.com"'
            )

        if not self.repo_path.exists():
            raise ValueError(f'Repository path does not exist: {self.repo_path}')

        if not (self.repo_path / '.git').exists():
            raise ValueError(f'Not a git repository: {self.repo_path}')

    def validate(self):
        """Check that repository is properly configured."""
        try:
            self.run_git(['rev-parse', '--is-inside-work-tree'], capture=True)
        except Exception as e:
            raise ValueError(f'Invalid git repository: {e}')

    def run_git(self, args: List[str], capture: bool = False) -> Optional[str]:
        """
        Execute a git command in the repository.

        Args:
            args: Git command arguments (without 'git' prefix)
            capture: If True, return stdout; if False, return None

        Returns:
            Command output if capture=True, None otherwise
        """
        env = os.environ.copy()
        env['GIT_AUTHOR_NAME'] = self.author_name
        env['GIT_AUTHOR_EMAIL'] = self.author_email
        env['GIT_COMMITTER_NAME'] = self.author_name
        env['GIT_COMMITTER_EMAIL'] = self.author_email

        cmd = ['git', '-C', str(self.repo_path)] + args
        logger.debug(f'Running: {" ".join(cmd)}')

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                timeout=30
            )
            return result.stdout.strip() if capture else None
        except subprocess.CalledProcessError as e:
            logger.error(f'Git command failed: {" ".join(cmd)}')
            logger.error(f'stdout: {e.stdout}')
            logger.error(f'stderr: {e.stderr}')
            raise


class SensorDataGenerator:
    """Generates realistic simulated sensor readings."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    @staticmethod
    def generate_readings(count: int = 10) -> List[Dict]:
        """
        Generate simulated sensor readings.

        Args:
            count: Number of readings to generate

        Returns:
            List of sensor reading dictionaries
        """
        readings = []
        base_temp = 20.0

        for i in range(count):
            readings.append({
                'timestamp': (datetime.now() - timedelta(seconds=i*30)).isoformat(),
                'temperature_c': round(base_temp + random.uniform(-2, 2), 1),
                'humidity_percent': round(random.uniform(30, 80), 1),
                'pressure_hpa': round(1013.25 + random.uniform(-5, 5), 2),
                'light_lux': round(random.uniform(100, 1000), 0),
                'motion_detected': random.choice([True, False])
            })

        return readings


class CommitManager:
    """Manages file creation and Git commits with duplicate protection."""

    def __init__(self, git_config: GitConfig):
        self.git_config = git_config
        self.data_dir = git_config.repo_path / 'sensor_data'
        self.hash_file = git_config.repo_path / '.demo_commit_hashes'
        self.committed_hashes = self._load_commit_hashes()

    def _load_commit_hashes(self) -> set:
        """Load the set of previously committed data hashes."""
        if self.hash_file.exists():
            try:
                with open(self.hash_file) as f:
                    return set(line.strip() for line in f if line.strip())
            except Exception as e:
                logger.warning(f'Could not load commit hashes: {e}')
        return set()

    def _save_commit_hash(self, data_hash: str):
        """Record a data hash as committed."""
        with open(self.hash_file, 'a') as f:
            f.write(data_hash + '\n')

    @staticmethod
    def _hash_data(data: Dict) -> str:
        """Generate a hash of data content for duplicate detection."""
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    def create_data_file(self, readings: List[Dict], timestamp: datetime) -> Optional[Path]:
        """
        Create a data file from readings.

        Returns:
            Path to created file, or None if duplicate detected
        """
        data = {
            'batch_id': timestamp.strftime('%Y%m%d_%H%M%S'),
            'readings': readings,
            'count': len(readings)
        }

        data_hash = self._hash_data(data)

        if data_hash in self.committed_hashes:
            logger.info(f'Skipping duplicate data batch (hash: {data_hash})')
            return None

        self.data_dir.mkdir(exist_ok=True)

        filename = f'{timestamp.strftime("%Y%m%d_%H%M%S")}_readings.json'
        filepath = self.data_dir / filename

        # Ensure unique filename
        counter = 1
        while filepath.exists():
            base = timestamp.strftime('%Y%m%d_%H%M%S')
            filename = f'{base}_{counter:02d}_readings.json'
            filepath = self.data_dir / filename
            counter += 1

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f'Created data file: {filepath.name} (hash: {data_hash})')
        return filepath, data_hash

    def commit_file(self, filepath: Path, timestamp: datetime):
        """Commit a data file with a descriptive message."""
        time_str = timestamp.strftime('%H:%M')
        message = f'data: record simulated reading at {time_str}'

        try:
            self.git_config.run_git(['add', str(filepath)])
            self.git_config.run_git(['commit', '-m', message])
            logger.info(f'Committed: {message}')
            return True
        except Exception as e:
            logger.error(f'Failed to commit {filepath.name}: {e}')
            # Restore file if commit failed
            try:
                self.git_config.run_git(['restore', '--staged', str(filepath)])
                filepath.unlink()
            except Exception as restore_err:
                logger.error(f'Could not clean up after failed commit: {restore_err}')
            return False

    def push_commits(self) -> bool:
        """Push commits to remote."""
        try:
            self.git_config.run_git(['push'])
            logger.info('Successfully pushed commits')
            return True
        except Exception as e:
            logger.error(f'Failed to push commits: {e}')
            logger.error('Push failed. Check remote configuration and authentication.')
            return False


class DemoOrchestrator:
    """Orchestrates the entire demo data generation and commit process."""

    def __init__(self, git_config: GitConfig, dry_run: bool = False):
        self.git_config = git_config
        self.dry_run = dry_run
        self.git_config.dry_run = dry_run
        self.sensor_gen = SensorDataGenerator()
        self.commit_mgr = CommitManager(git_config)

    def should_run_today(self) -> bool:
        """
        Check if data should be generated today.

        In a production demo, you might skip certain days or conditions.
        This is a placeholder for more sophisticated scheduling logic.
        """
        return True

    def run(self) -> bool:
        """
        Execute a complete demo run: generate data, commit, and push.

        Returns:
            True if successful, False otherwise
        """
        logger.info('Starting GitHub activity demo run')
        logger.info(f'Repository: {self.git_config.repo_path}')
        logger.info(f'Author: {self.git_config.author_name} <{self.git_config.author_email}>')

        if self.dry_run:
            logger.info('DRY RUN MODE: No actual commits or pushes will be made')

        try:
            self.git_config.validate()
        except ValueError as e:
            logger.error(f'Git validation failed: {e}')
            return False

        if not self.should_run_today():
            logger.info('Skipping run based on scheduling logic')
            return True

        # Generate 1-5 random batches
        num_batches = random.randint(1, 5)
        logger.info(f'Generating {num_batches} data batches')

        successful_commits = 0
        now = datetime.now()

        for batch_num in range(num_batches):
            # Generate readings for this batch
            num_readings = random.randint(5, 15)
            readings = self.sensor_gen.generate_readings(num_readings)

            # Stagger timestamps slightly within the current day
            batch_time = now - timedelta(
                minutes=random.randint(0, 480),  # Within last 8 hours
                seconds=random.randint(0, 59)
            )

            if self.dry_run:
                logger.info(f'[DRY RUN] Would create batch {batch_num + 1} with {num_readings} readings')
                logger.info(f'[DRY RUN] Timestamp: {batch_time.isoformat()}')
                logger.info(f'[DRY RUN] Sample reading: {readings[0]}')
                successful_commits += 1
            else:
                result = self.commit_mgr.create_data_file(readings, batch_time)
                if result:
                    filepath, data_hash = result
                    if self.commit_mgr.commit_file(filepath, batch_time):
                        self.commit_mgr._save_commit_hash(data_hash)
                        successful_commits += 1

        logger.info(f'Successfully committed {successful_commits}/{num_batches} batches')

        if successful_commits == 0:
            logger.info('No new data committed')
            return True

        if self.dry_run:
            logger.info('[DRY RUN] Would push commits now')
            return True

        # Push all commits
        if not self.commit_mgr.push_commits():
            logger.warning('Commits created locally but push failed')
            return False

        logger.info('GitHub activity demo run completed successfully')
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate simulated sensor data and commit to GitHub for demonstration'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would happen without making actual commits'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only validate configuration without running'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        git_config = GitConfig()

        if args.check_only:
            git_config.validate()
            logger.info('Configuration is valid')
            return 0

        orchestrator = DemoOrchestrator(git_config, dry_run=args.dry_run)
        success = orchestrator.run()
        return 0 if success else 1

    except Exception as e:
        logger.error(f'Fatal error: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
