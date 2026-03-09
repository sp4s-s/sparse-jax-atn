import subprocess
import sys


def main():
    exit_code = subprocess.call(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
