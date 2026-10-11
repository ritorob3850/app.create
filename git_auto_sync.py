"""
GitHub Auto-Sync Engine for app.create
======================================
Monitors the folder for any file additions, edits, or deletions.
Whenever changes occur, it automatically stages, commits, and pushes to GitHub.

Features:
- Zero external dependencies (uses standard Python library)
- Debounce protection (waits for saves to settle before pushing)
- Automatic branch detection
- Graceful error handling (keeps running even if internet temporarily drops)
"""

import os
import sys
import time
import subprocess
from datetime import datetime

# Configuration
POLL_INTERVAL_SECONDS = 3     # Check for file changes every 3 seconds
DEBOUNCE_SECONDS = 2          # Wait 2 seconds after detecting change before pushing
REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def run_git(args: list[str]) -> tuple[int, str]:
    """Runs a git command in the repository directory and returns (exit_code, output)."""
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode, proc.stdout.strip()
    except Exception as e:
        return 1, str(e)


def get_current_branch() -> str:
    """Gets the active git branch name."""
    code, out = run_git(["branch", "--show-current"])
    if code == 0 and out:
        return out.strip()
    return "master"


def get_changed_files_summary(porcelain_output: str) -> str:
    """Summarizes changed files into a short string for the commit message."""
    lines = [line.strip() for line in porcelain_output.splitlines() if line.strip()]
    if not lines:
        return "updates"
    
    file_names = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            file_names.append(parts[1])
        else:
            file_names.append(line)
            
    if len(file_names) == 1:
        return file_names[0]
    elif len(file_names) <= 3:
        return ", ".join(file_names)
    else:
        return f"{file_names[0]}, {file_names[1]} and {len(file_names) - 2} more"


def main():
    print("=" * 60)
    print("  🚀 GITHUB AUTO-SYNC ENGINE — app.create")
    print("=" * 60)
    print(f"📁 Monitoring: {REPO_DIR}")

    branch = get_current_branch()
    print(f"🌿 Target Branch: {branch}")

    code, remotes = run_git(["remote", "-v"])
    if code != 0 or not remotes:
        print("❌ Error: No Git remote configured in this repository.")
        input("Press Enter to exit...")
        return

    print("📡 Remote configured successfully.")
    print(f"⏱️  Checking every {POLL_INTERVAL_SECONDS}s. Press Ctrl+C at any time to stop.")
    print("-" * 60)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🟢 Watching for changes...")

    last_had_changes = False

    while True:
        try:
            # Check git status
            code, status = run_git(["status", "--porcelain"])
            
            if code == 0 and status:
                now_str = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{now_str}] 📝 Change detected! Waiting {DEBOUNCE_SECONDS}s for edits to settle...")
                time.sleep(DEBOUNCE_SECONDS)

                # Re-check status in case more edits occurred
                code, status_now = run_git(["status", "--porcelain"])
                if not status_now:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ Changes reverted or empty, resuming watch.")
                    continue

                summary = get_changed_files_summary(status_now)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                commit_msg = f"Auto-update: {timestamp} [{summary}]"

                print(f"[{datetime.now().strftime('%H:%M:%S')}] ➕ Staging changed files...")
                run_git(["add", "-A"])

                print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Committing: '{commit_msg}'")
                c_code, c_out = run_git(["commit", "-m", commit_msg])
                if c_code != 0 and "nothing to commit" not in c_out.lower():
                    print(f"⚠️ Commit warning: {c_out}")

                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Pushing to GitHub (origin/{branch})...")
                p_code, p_out = run_git(["push", "origin", branch])

                if p_code == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ PUSH SUCCESSFUL! GitHub is now fully updated.")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Push failed (might be offline or temporary conflict):")
                    print(f"   {p_out}")
                    print("   Will retry automatically on next detected change.")

                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🟢 Continuing to watch for new changes...")
                last_had_changes = True
            else:
                last_had_changes = False

            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\n\n🛑 Auto-sync stopped by user. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
