#!/usr/bin/env python3
"""
Script to publish LLM Studio to PyPI.
Handles building, testing, and uploading the package.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if check and result.returncode != 0:
        sys.exit(f"Command failed: {cmd}")

    return result


def clean_build_artifacts():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")

    artifacts = ["build", "dist", "*.egg-info"]
    for pattern in artifacts:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"   Removed directory: {path}")
            else:
                path.unlink()
                print(f"   Removed file: {path}")


def run_tests():
    """Run the test suite to ensure everything works."""
    print("🧪 Running test suite...")
    result = run_command("python -m pytest tests/ -v", check=False)

    if result.returncode != 0:
        print("❌ Tests failed! Fix issues before publishing.")
        sys.exit(1)

    print("✅ All tests passed!")


def build_package():
    """Build the package for distribution."""
    print("📦 Building package...")
    run_command("python -m build")
    print("✅ Package built successfully!")


def check_package():
    """Check the package with twine."""
    print("🔍 Checking package...")
    run_command("python -m twine check dist/*")
    print("✅ Package check passed!")


def upload_to_test_pypi():
    """Upload to Test PyPI first."""
    print("🚀 Uploading to Test PyPI...")
    run_command("python -m twine upload --repository testpypi dist/*")
    print("✅ Uploaded to Test PyPI!")
    print("🔗 Check at: https://test.pypi.org/project/llm-studio/")


def upload_to_pypi():
    """Upload to production PyPI."""
    print("🚀 Uploading to PyPI...")
    response = input("Are you sure you want to upload to production PyPI? (y/N): ")

    if response.lower() != "y":
        print("❌ Upload cancelled.")
        return

    run_command("python -m twine upload dist/*")
    print("✅ Uploaded to PyPI!")
    print("🔗 Check at: https://pypi.org/project/llm-studio/")


def main():
    """Main publishing workflow."""
    print("🚀 LLM Studio PyPI Publishing Script")
    print("=" * 40)

    # Ensure we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Run from project root.")
        sys.exit(1)

    # Install build dependencies
    print("📋 Installing build dependencies...")
    run_command("pip install build twine")

    # Clean previous builds
    clean_build_artifacts()

    # Run tests
    run_tests()

    # Build package
    build_package()

    # Check package
    check_package()

    # Upload options
    print("\n🎯 Upload Options:")
    print("1. Upload to Test PyPI (recommended first)")
    print("2. Upload to Production PyPI")
    print("3. Exit")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        upload_to_test_pypi()
    elif choice == "2":
        upload_to_pypi()
    else:
        print("👋 Build complete. Package ready in dist/")

    print("\n✅ Publishing script complete!")


if __name__ == "__main__":
    main()
