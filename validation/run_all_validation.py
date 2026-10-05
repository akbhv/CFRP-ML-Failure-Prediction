import subprocess
import sys


VALIDATION_MODULES = [
    "validation.test_lamina",
    "validation.test_Q",
    "validation.test_transformation",
    "validation.test_clt",
    "validation.test_failure_criteria",
    "validation.test_laminate_failure_comparison",
    "validation.test_laminate_failure_envelope",
    "validation.test_laminate_loading_study",
]


def main():
    print("=" * 90)
    print("COMPLETE CFRP PHYSICS VALIDATION")
    print("=" * 90)
    print()

    passed = []

    for module in VALIDATION_MODULES:

        print("=" * 90)
        print(f"RUNNING: {module}")
        print("=" * 90)

        result = subprocess.run(
            [sys.executable, "-m", module]
        )

        if result.returncode != 0:
            print()
            print("=" * 90)
            print(f"VALIDATION FAILED: {module}")
            print("=" * 90)

            print()
            print("Modules passed before failure:")
            for passed_module in passed:
                print(f"  PASS  {passed_module}")

            sys.exit(result.returncode)

        passed.append(module)

        print()
        print(f"PASSED: {module}")
        print()

    print("=" * 90)
    print("VALIDATION SUMMARY")
    print("=" * 90)

    for module in passed:
        print(f"PASS  {module}")

    print()
    print("=" * 90)
    print("COMPLETE PHYSICS VALIDATION PASSED")
    print("=" * 90)


if __name__ == "__main__":
    main()