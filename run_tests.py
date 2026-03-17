import os
import subprocess
import sys

def run_test(test_file):
    print(f"Running {test_file}...", end=" ", flush=True)
    try:
        basename = os.path.basename(test_file)
        if basename.startswith("test_repl_"):
            with open(test_file, "r", encoding="utf-8") as f:
                result = subprocess.run(
                    ["./tensorpy"],
                    input=f.read(),
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            if result.returncode == 0 and "3" in result.stdout:
                print("\033[92mPASSED\033[0m")
                return True, result.stdout
            print("\033[91mFAILED\033[0m")
            return False, result.stderr + result.stdout

        # Run with a timeout to prevent infinite loops
        result = subprocess.run(
            ["./tensorpy", test_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        expected_rc = 0
        if basename.startswith("test_error"):
            expected_rc = 65
        if basename.startswith("test_runtime_error"):
            expected_rc = 70

        if result.returncode == expected_rc:
            print("\033[92mPASSED\033[0m")
            return True, result.stdout
        else:
            print("\033[91mFAILED\033[0m")
            return False, result.stderr + result.stdout
    except subprocess.TimeoutExpired:
        print("\033[91mTIMEOUT\033[0m")
        return False, "Test timed out"
    except Exception as e:
        print("\033[91mERROR\033[0m")
        return False, str(e)

def main():
    test_dir = "tests"
    if not os.path.exists("./tensorpy"):
        print("Error: tensorpy executable not found. Run 'make' first.")
        sys.exit(1)

    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".py")]
    test_files.sort()

    passed_count = 0
    failed_tests = []

    print("-" * 40)
    print(f"Found {len(test_files)} test files.")
    print("-" * 40)

    for test_file in test_files:
        passed, output = run_test(test_file)
        if passed:
            passed_count += 1
        else:
            failed_tests.append((test_file, output))

    print("-" * 40)
    print(f"Summary: {passed_count}/{len(test_files)} passed")
    print("-" * 40)

    if failed_tests:
        print("\nFailed Tests Details:")
        for test_file, output in failed_tests:
            print(f"\n--- {test_file} ---")
            # Print last 10 lines of output
            lines = output.strip().split("\n")
            for line in lines[-10:]:
                print(line)
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
