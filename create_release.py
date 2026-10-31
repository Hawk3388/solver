"""Build solver.exe with solver.spec and package it with the latest local detector."""

import argparse
import hashlib
from importlib.util import find_spec
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


def build_executable(project_root):
    """Only replace dist/solver.exe after a successful, isolated PyInstaller build."""
    spec = project_root / "solver.spec"
    if not spec.is_file():
        raise FileNotFoundError(f"Build configuration is missing: {spec}")
    if find_spec("PyInstaller") is None:
        raise ModuleNotFoundError(
            f"PyInstaller is not installed for {sys.executable}. "
            "Install the project requirements and run: python -m pip install pyinstaller pyinstaller-hooks-contrib"
        )
    build_directory = project_root / "build"
    build_directory.mkdir(parents=True, exist_ok=True)
    destination = project_root / "dist" / "solver.exe"
    print(f"Building solver.exe with {sys.executable} and {spec}...", flush=True)
    with tempfile.TemporaryDirectory(prefix="release-build-", dir=build_directory) as staging:
        subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm",
             "--distpath", staging, "--workpath", str(build_directory), str(spec)],
            cwd=project_root,
            check=True,
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
        print(f"PyInstaller failed with exit code {error.returncode}. The release ZIP was not updated.", file=sys.stderr)
        return 1
    except (OSError, ValueError, zipfile.BadZipFile, ImportError) as error:
        print(f"Release packaging failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
