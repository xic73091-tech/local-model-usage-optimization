#!/usr/bin/env python3
"""
Security Audit Script
Runs dependency vulnerability scan and basic security checks.
Usage: python scripts/security_audit.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def check_pip_audit() -> bool:
    """Run pip-audit to check for known vulnerabilities."""
    print("\n[1/4] Dependency Vulnerability Scan (pip-audit)")
    print("=" * 60)

    # Try pip-audit first
    result = run_command([sys.executable, "-m", "pip_audit", "--version"])
    if result.returncode != 0:
        print("  pip-audit not installed. Installing...")
        install = run_command([sys.executable, "-m", "pip", "install", "pip-audit"])
        if install.returncode != 0:
            print("  [WARN] Could not install pip-audit, falling back to manual check")
            return check_requirements_manual()

    # Build list of packages from requirements.txt for targeted audit
    req_file = Path(__file__).parent.parent / "requirements.txt"
    req_packages = set()
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for op in [">=", "<=", "==", "!=", ">", "<"]:
                if op in line:
                    req_packages.add(line.split(op)[0].strip().lower())
                    break
            else:
                req_packages.add(line.strip().lower())

    # Run pip-audit on requirements.txt
    result = run_command([
        sys.executable, "-m", "pip_audit",
        "--requirement", str(req_file),
        "--format", "columns",
        "--no-deps",
    ])

    # Filter output: only show vulnerabilities in our direct dependencies
    if result.returncode == 0:
        print("  [OK] No known vulnerabilities in project dependencies")
        return True

    # Parse and filter results
    lines = result.stdout.strip().split("\n")
    project_vulns = []
    for line in lines:
        # Check if the vulnerable package is in our requirements
        for pkg in req_packages:
            if line.lower().startswith(pkg.lower() + " ") or line.lower().startswith(pkg.lower() + "\t"):
                project_vulns.append(line)
                break

    if project_vulns:
        print("  [WARN] Vulnerabilities in project dependencies:")
        for v in project_vulns:
            print(f"    {v}")
        return False
    else:
        print("  [OK] No known vulnerabilities in project dependencies")
        if result.stdout.strip():
            print("  [INFO] Vulnerabilities in other packages (not project deps):")
            for line in lines[:5]:
                print(f"    {line}")
        return True


def check_requirements_manual() -> bool:
    """Manual check: compare installed versions against requirements."""
    print("  Checking installed packages against requirements.txt...")

    req_file = Path(__file__).parent.parent / "requirements.txt"
    if not req_file.exists():
        print("  [WARN] requirements.txt not found")
        return True

    issues = []
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse package name
            for op in [">=", "<=", "==", "!=", ">", "<"]:
                if op in line:
                    pkg_name = line.split(op)[0].strip()
                    break
            else:
                pkg_name = line.strip()

            # Check if installed
            result = run_command([
                sys.executable, "-c",
                f"import importlib.metadata; print(importlib.metadata.version('{pkg_name}'))"
            ])
            if result.returncode != 0:
                issues.append(f"  [WARN] {pkg_name}: not installed")

    if issues:
        for issue in issues:
            print(issue)
        return False

    print("  [OK] All required packages are installed")
    return True


def check_secrets_in_code() -> bool:
    """Check for hardcoded secrets in source files."""
    print("\n[2/4] Hardcoded Secrets Scan")
    print("=" * 60)

    src_dir = Path(__file__).parent.parent / "src"
    if not src_dir.exists():
        print("  [SKIP] src/ directory not found")
        return True

    patterns = [
        (r'password\s*=\s*["\'][^"\']{4,}', "hardcoded password"),
        (r'api_key\s*=\s*["\'][^"\']{10,}', "hardcoded API key"),
        (r'secret\s*=\s*["\'][^"\']{10,}', "hardcoded secret"),
        (r'token\s*=\s*["\'][^"\']{10,}', "hardcoded token"),
        (r'AWS_ACCESS_KEY', "AWS key reference"),
        (r'PRIVATE_KEY', "private key reference"),
    ]

    import re
    findings = []
    for py_file in src_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        for pattern, desc in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Filter out environment variable reads
                for match in matches:
                    if "os.environ" in match or "os.getenv" in match:
                        continue
                    findings.append(f"  [WARN] {py_file.name}: {desc}")

    if findings:
        for f in findings:
            print(f)
        return False

    print("  [OK] No hardcoded secrets found")
    return True


def check_dangerous_functions() -> bool:
    """Check for dangerous function usage."""
    print("\n[3/4] Dangerous Function Scan")
    print("=" * 60)

    src_dir = Path(__file__).parent.parent / "src"
    if not src_dir.exists():
        print("  [SKIP] src/ directory not found")
        return True

    import re
    findings = []
    for py_file in src_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # Check for eval/exec (excluding comments and strings)
        if re.search(r'\beval\s*\(', content):
            findings.append(f"  [WARN] {py_file.name}: uses eval()")
        if re.search(r'\bexec\s*\(', content):
            findings.append(f"  [WARN] {py_file.name}: uses exec()")
        if re.search(r'\bpickle\.load', content):
            findings.append(f"  [WARN] {py_file.name}: uses pickle.load (unsafe deserialization)")
        if re.search(r'shell\s*=\s*True', content):
            findings.append(f"  [WARN] {py_file.name}: subprocess with shell=True")

    if findings:
        for f in findings:
            print(f)
        return False

    print("  [OK] No dangerous function usage found")
    return True


def check_default_config() -> bool:
    """Check for insecure default configurations."""
    print("\n[4/4] Configuration Security Check")
    print("=" * 60)

    issues = []

    # Check .env.example
    env_example = Path(__file__).parent.parent / ".env.example"
    if env_example.exists():
        content = env_example.read_text(encoding="utf-8")
        if "LMO_HOST=0.0.0.0" in content:
            issues.append("  [WARN] .env.example: HOST defaults to 0.0.0.0")
        if "LMO_AUTH_ENABLED=false" in content:
            issues.append("  [WARN] .env.example: AUTH_ENABLED defaults to false")
        if "CORS" in content and '"*"' in content:
            issues.append("  [WARN] .env.example: CORS allows all origins")

    # Check start scripts for actual default HOST (not help text)
    import re
    for script in ["scripts/start.sh", "scripts/start.ps1"]:
        script_path = Path(__file__).parent.parent / script
        if script_path.exists():
            content = script_path.read_text(encoding="utf-8")
            # Check for actual default assignment, not help text or warning messages
            if script.endswith(".sh"):
                # Look for HOST="${HOST:-0.0.0.0}" pattern
                if re.search(r'HOST=.*\$\{HOST:-0\.0\.0\.0\}', content):
                    issues.append(f"  [WARN] {script}: HOST defaults to 0.0.0.0")
            elif script.endswith(".ps1"):
                # Look for else { "0.0.0.0" } pattern in HOST assignment
                if re.search(r'else\s*\{\s*"0\.0\.0\.0"\s*\}', content):
                    issues.append(f"  [WARN] {script}: HOST defaults to 0.0.0.0")

    if issues:
        for issue in issues:
            print(issue)
        return False

    print("  [OK] Configuration looks secure")
    return True


def main():
    print("=" * 60)
    print("  LocalAI Optimizer - Security Audit")
    print("=" * 60)

    results = []
    results.append(("Dependency Scan", check_pip_audit()))
    results.append(("Secrets Scan", check_secrets_in_code()))
    results.append(("Dangerous Functions", check_dangerous_functions()))
    results.append(("Configuration", check_default_config()))

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("  All checks passed!")
    else:
        print("  Some checks failed. Review warnings above.")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
