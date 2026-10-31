"""Build solver.exe with solver.spec and package it with the latest local detector."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILENAME = "gap_detection_model.pt"
VERSION_PATTERN = re.compile(r"v(\d+)\.(\d+)\.(\d+)")


def build_environment(project_root):
    """Prepare a private venv instead of freezing the caller's development environment."""
    build_directory = project_root / "build"
    environment = build_directory / "release-env"
    marker = build_directory / "release-env.json"
    requirements = project_root / "requirements-build.txt"
    runtime_requirements = project_root / "requirements.txt"
    for path in (requirements, runtime_requirements):
        if not path.is_file():
            raise FileNotFoundError(f"Release requirements are missing: {path}")
    fingerprint = hashlib.sha256(
        b"cpu-release-env-v1\n" + requirements.read_bytes() + runtime_requirements.read_bytes()
    ).hexdigest()
    if environment.exists() and not marker.is_file():
        raise ValueError(f"Unmanaged build environment at {environment}. Move it before running the release script.")
    state = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
    if state.get("environment", str(environment)) != str(environment):
        raise ValueError(f"Unexpected build environment path in {marker}.")
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    process_environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        process_environment.pop(name, None)
    process_environment["PYTHONNOUSERSITE"] = "1"
    process_environment["PYTHONUTF8"] = "1"
    build_directory.mkdir(parents=True, exist_ok=True)
    settings_directory = build_directory / "ultralytics-settings"
    settings_directory.mkdir(exist_ok=True)
    process_environment["YOLO_CONFIG_DIR"] = str(settings_directory)
    process_environment["YOLO_AUTOINSTALL"] = "false"

    if not python.is_file() or state.get("fingerprint") != fingerprint:
        marker.write_text(json.dumps({"environment": str(environment), "status": "preparing"}), encoding="utf-8")
        if not python.is_file():
            print(f"Creating isolated release environment: {environment}", flush=True)
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True, env=process_environment)
        pip = [str(python), "-m", "pip", "--disable-pip-version-check"]
        subprocess.run(pip + ["install", "--upgrade", "pip"], check=True, env=process_environment)
        cpu_packages = [line.strip() for line in requirements.read_text(encoding="utf-8").splitlines()
                        if line.startswith(("torch==", "torchvision=="))]
        if len(cpu_packages) != 2 or any("+cpu" not in package for package in cpu_packages):
            raise ValueError("requirements-build.txt must pin CPU-only torch and torchvision wheels.")
        print("Installing CPU-only PyTorch for the release build...", flush=True)
        subprocess.run(pip + ["install", "--upgrade", "--force-reinstall", "--index-url",
                             "https://download.pytorch.org/whl/cpu", *cpu_packages],
                       check=True, env=process_environment)
        subprocess.run(pip + ["install", "-r", str(requirements)], cwd=project_root,
                       check=True, env=process_environment)
        subprocess.run(pip + ["check"], check=True, env=process_environment)
    # Check on every build, including cached environments.
    verification = (
        "import sys, pathlib, torch, torchvision; "
        "assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve(), 'Wrong build interpreter'; "
        "assert sys.prefix != sys.base_prefix, 'Build requires an isolated virtual environment'; "
        "assert torch.version.cuda is None and getattr(torch.version, 'hip', None) is None, 'GPU PyTorch is not allowed'; "
        "assert '+cpu' in torch.__version__ and '+cpu' in torchvision.__version__, 'CPU wheels are required'; "
        "print('Verified CPU build: torch', torch.__version__, '/ torchvision', torchvision.__version__)"
    )
    subprocess.run([str(python), "-c", verification, str(environment)], check=True, env=process_environment)
    marker.write_text(json.dumps({"environment": str(environment), "fingerprint": fingerprint}), encoding="utf-8")
    return python, process_environment


def build_executable(project_root):
    """Only replace dist/solver.exe after a successful, isolated PyInstaller build."""
    spec = project_root / "solver.spec"
    if not spec.is_file():
        raise FileNotFoundError(f"Build configuration is missing: {spec}")
    python, process_environment = build_environment(project_root)
    build_directory = project_root / "build"
    build_directory.mkdir(parents=True, exist_ok=True)
    destination = project_root / "dist" / "solver.exe"
    print(f"Building app.py with CPU-only environment {python} and {spec}...", flush=True)
    with tempfile.TemporaryDirectory(prefix="release-build-", dir=build_directory) as staging:
        subprocess.run(
            [str(python), "-m", "PyInstaller", "--clean", "--noconfirm",
             "--distpath", staging, "--workpath", str(build_directory), str(spec)],
            cwd=project_root,
            check=True,
            env=process_environment,
        )
        executable = Path(staging) / "solver.exe"
        if not executable.is_file() or not executable.stat().st_size:
            raise FileNotFoundError("PyInstaller finished without creating a non-empty solver.exe.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        executable.replace(destination)
    return destination


def latest_model(project_root):
    """Select model/vX.Y.Z numerically, independently of application tags."""
    model_directory = project_root / "model"
    candidates = []
    if model_directory.is_dir():
        for directory in model_directory.iterdir():
            version = VERSION_PATTERN.fullmatch(directory.name)
            weights = directory / MODEL_FILENAME
            if directory.is_dir() and version and weights.is_file():
                candidates.append((tuple(map(int, version.groups())), weights))
    if not candidates:
        raise FileNotFoundError(
            f"No detector weights found. Expected {model_directory}/vX.Y.Z/{MODEL_FILENAME}."
        )
    weights = max(candidates, key=lambda candidate: candidate[0])[1]
    if not weights.stat().st_size:
        raise ValueError(f"The newest detector file is empty: {weights}")
    return weights


def release_files(project_root, executable=None):
    """Return an explicit allowlist; never copy private configuration or datasets."""
    if executable is None:
        executable = project_root / "dist" / "solver.exe"
        if not executable.is_file():
            executable = project_root / "solver" / "solver.exe"
    if not executable.is_file() or not executable.stat().st_size:
        raise FileNotFoundError(
            "No built executable found. Build solver.spec first or pass --exe PATH. "
            f"Checked executable: {executable}"
        )
    weights = latest_model(project_root)
    files = [(executable, "solver.exe")]
    for name in ("README.md", "LICENSE.md", ".env.example"):
        path = project_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required release file is missing: {path}")
        files.append((path, name))
    files.append((weights, f"model/{weights.parent.name}/{MODEL_FILENAME}"))
    return files


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_release(project_root=PROJECT_ROOT, *, executable=None, output=None, force=False, skip_build=False):
    """Build and verify an archive without modifying the existing solver folder."""
    project_root = Path(project_root).resolve()
    executable = Path(executable).resolve() if executable is not None else None
    output = Path(output).resolve() if output is not None else project_root / "dist" / "solver.zip"
    if output.suffix.lower() != ".zip":
        raise ValueError("The output filename must end in .zip.")
    checksum = output.with_name(output.name + ".sha256")
    for path in (output, checksum):
        if path.exists() and (not force or not path.is_file()):
            raise FileExistsError(f"Output already exists: {path}. Use --force to replace existing files.")
    if executable is not None and not skip_build:
        raise ValueError("--exe requires --skip-build. The default build always uses the freshly built executable.")
    # Fail before a lengthy build if packaging inputs are missing.
    latest_model(project_root)
    for name in ("README.md", "LICENSE.md", ".env.example"):
        if not (project_root / name).is_file():
            raise FileNotFoundError(f"Required release file is missing: {project_root / name}")
    if not skip_build:
        executable = build_executable(project_root)
    files = release_files(project_root, executable)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="solver-release-", suffix=".zip.tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    checksum_temporary = None
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True, strict_timestamps=False) as archive:
            for source, name in files:
                print(f"Adding {name} <- {source}", flush=True)
                archive.write(source, arcname=name)
        print("Verifying archive...", flush=True)
        with zipfile.ZipFile(temporary) as archive:
            damaged = archive.testzip()
            if damaged:
                raise OSError(f"ZIP verification failed for {damaged}.")
        checksum_text = f"{sha256(temporary)}  {output.name}\n"
        descriptor, checksum_name = tempfile.mkstemp(prefix="solver-release-", suffix=".sha256.tmp", dir=output.parent)
        checksum_temporary = Path(checksum_name)
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as destination:
            destination.write(checksum_text)
        if not force and (output.exists() or checksum.exists()):
            raise FileExistsError("Output appeared during packaging. Use --force to replace it.")
        temporary.replace(output)
        checksum_temporary.replace(checksum)
    finally:
        temporary.unlink(missing_ok=True)
        if checksum_temporary is not None:
            checksum_temporary.unlink(missing_ok=True)
    print(f"Created: {output} ({output.stat().st_size / (1024 * 1024):.1f} MiB)")
    print(f"SHA-256: {checksum}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="Package an existing executable instead of rebuilding it.")
    parser.add_argument("--exe", type=Path, help="Existing executable to package; requires --skip-build.")
    parser.add_argument("--output", type=Path, help="ZIP destination; defaults to dist/solver.zip beside this script.")
    parser.add_argument("--force", action="store_true", help="Replace an existing ZIP and checksum.")
    args = parser.parse_args(argv)
    if args.exe is not None and not args.skip_build:
        parser.error("--exe requires --skip-build")
    try:
        create_release(executable=args.exe, output=args.output, force=args.force, skip_build=args.skip_build)
    except subprocess.CalledProcessError as error:
        print(f"Release environment setup or build failed with exit code {error.returncode}. The release ZIP was not updated.", file=sys.stderr)
        return 1
    except (OSError, ValueError, zipfile.BadZipFile, ImportError) as error:
        print(f"Release packaging failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
