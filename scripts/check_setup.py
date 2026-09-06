"""Check that the lab is set up correctly.

Run it once after the setup exercise, and again any time something feels off:

    uv run python scripts/check_setup.py

Every check prints one line. A FAIL line always comes with one sentence
telling you what to do about it. The script only reads: it never changes
anything on your machine.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PASS = "  PASS"
FAIL = "  FAIL"
WARN = "  NOTE"

failures = 0


def ok(message):
    print(f"{PASS}  {message}")


def bad(message, fix):
    global failures
    failures += 1
    print(f"{FAIL}  {message}")
    print(f"        Fix: {fix}")


def note(message):
    print(f"{WARN}  {message}")


def check_python():
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info[:2] == (3, 11):
        ok(f"Python {version}")
    else:
        bad(
            f"Python is {version}, the lab needs 3.11",
            "install Python 3.11 and run uv sync again, uv will pick it up",
        )


def check_uv():
    if shutil.which("uv"):
        ok("uv is installed")
    else:
        bad(
            "uv is not on your PATH",
            "install it from https://docs.astral.sh/uv/ and reopen your terminal",
        )


def docker_info():
    try:
        raw = subprocess.run(
            ["docker", "info", "--format", "{{.MemTotal}}"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if raw.returncode != 0:
        return None
    try:
        return int(raw.stdout.strip())
    except ValueError:
        return None


def check_docker():
    mem = docker_info()
    if mem is None:
        bad(
            "Docker is not running or not installed",
            "start Docker Desktop (or the Docker daemon) and run this check again",
        )
        return False
    gb = mem / (1024 ** 3)
    if gb >= 5.5:
        ok(f"Docker is running with {gb:.1f} GB of memory")
    else:
        bad(
            f"Docker has only {gb:.1f} GB of memory, the lab needs at least 6",
            "raise the memory limit in Docker Desktop settings, then restart Docker",
        )
    return True


def compose_containers():
    """Return this project's compose containers as a list of dicts."""
    try:
        raw = subprocess.run(
            ["docker", "compose", "ps", "-a", "--format", "json"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if raw.returncode != 0:
        return []
    containers = []
    for line in raw.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                containers.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return containers


def published_ports(containers):
    ports = set()
    for c in containers:
        for pub in c.get("Publishers") or []:
            if pub.get("PublishedPort"):
                ports.add(int(pub["PublishedPort"]))
    return ports


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def check_ports(containers):
    ours = published_ports(containers)
    for port in (5001, 6379, 9090, 3001, 8081):
        if not port_in_use(port):
            ok(f"port {port} is free")
        elif port in ours:
            ok(f"port {port} is held by this project's containers")
        else:
            bad(
                f"port {port} is held by something that is not this lab",
                f"find it with `lsof -i :{port}` or `docker ps`, stop it, "
                "then run `docker compose up -d --wait` again",
            )
    # Port 8000 is different: it belongs to the serving API you build in the
    # serving exercise, which runs on the host, not in a container.
    if not port_in_use(8000):
        ok("port 8000 is free (your serving API will use it later)")
    else:
        note(
            "port 8000 is in use: fine if that is your own serving API, "
            "otherwise stop whatever holds it before the serving exercise"
        )


def check_services(containers):
    if not containers:
        bad(
            "no lab containers are running",
            "run `docker compose up -d --wait` from the repository root",
        )
        return
    for c in sorted(containers, key=lambda c: c.get("Service", "")):
        service = c.get("Service", "unknown")
        state = c.get("State", "")
        health = c.get("Health", "")
        if service == "airflow-init":
            # A one-shot task: it runs the database migration and exits.
            if state == "exited" and c.get("ExitCode", 1) == 0:
                ok("airflow-init completed (one-shot setup task)")
            else:
                bad(
                    f"airflow-init is {state} with exit code {c.get('ExitCode')}",
                    "read its logs with `docker compose logs airflow-init`",
                )
        elif state == "running" and health in ("healthy", ""):
            label = "healthy" if health == "healthy" else "running"
            ok(f"{service} is {label}")
        else:
            bad(
                f"{service} is {state} ({health or 'no health yet'})",
                f"read its logs with `docker compose logs {service} --tail 50`",
            )


def check_data_files():
    for path in (
        "data/reference.csv",
        "data/drifted.csv",
        "feature_repo/data/district_features.parquet",
    ):
        if Path(path).exists():
            ok(f"{path} exists")
        else:
            script = (
                "scripts/make_drift.py" if "drifted" in path else "scripts/seed_data.py"
            )
            bad(
                f"{path} is missing",
                f"run `uv run python {script}` from the repository root",
            )


def check_mlflow():
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        bad(
            "MLFLOW_TRACKING_URI is not set in this shell",
            "run `export MLFLOW_TRACKING_URI=http://localhost:5001` "
            "(and add it to your shell profile so new terminals have it)",
        )
        return
    ok(f"MLFLOW_TRACKING_URI is set to {uri}")
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=10) as resp:
            if resp.status == 200:
                ok("MLflow responds at that address")
            else:
                bad(
                    f"MLflow answered with status {resp.status}",
                    "read its logs with `docker compose logs mlflow --tail 50`",
                )
    except (urllib.error.URLError, OSError):
        bad(
            "nothing answered at that address",
            "start the services with `docker compose up -d --wait`, "
            "then run this check again",
        )


def main():
    print("Checking your lab setup\n")
    check_python()
    check_uv()
    docker_up = check_docker()
    containers = compose_containers() if docker_up else []
    if docker_up:
        check_ports(containers)
        check_services(containers)
    check_data_files()
    check_mlflow()

    print()
    if failures == 0:
        print("All checks passed. You are ready for the exercises.")
        return 0
    plural = "check" if failures == 1 else "checks"
    print(f"{failures} {plural} failed. Fix the lines above, then run this again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
