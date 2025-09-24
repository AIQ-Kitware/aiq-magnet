#!/usr/bin/env python3
"""
Download HELM benchmark run artifacts from the public GCS bucket.

Features:
  - Auto-detect latest version if version=auto (default)
  - Prompt to install gsutil if missing (Debian/Ubuntu auto-install supported)
  - Choose benchmark name (e.g., lite, helm)
  - Key/value args: dir=, version=, benchmark=, checksum=, install=
  - Flags: --list-benchmarks, --list-versions, --verbose
  - Idempotent using 'gsutil rsync' (optionally checksum mode)

Example:
    >>> from magnet.backends.helm import download_helm_results
    >>> import ubelt as ub
    >>> #
    >>> # Test listing benchamrks
    >>> with ub.CaptureStdout(suppress=False) as cap:
    >>>     download_helm_results.main(list_benchmarks=True)
    >>> assert len(cap.text.split()) >= 29
    >>> #
    >>> # Test listing versions
    >>> with ub.CaptureStdout(suppress=False) as cap:
    >>>     download_helm_results.main(list_versions=True, benchmark='lite')
    >>> assert len(cap.text.split()) >= 14

Example:
    >>> from magnet.backends.helm import download_helm_results
    >>> import ubelt as ub
    >>> # Start fresh
    >>> dpath = ub.Path.appdir('magnet/tests/download_helm_list')
    >>> dpath.delete()
    >>> assert len(list(dpath.glob('**'))) == 0, 'delete should remove everything'
    >>> #
    >>> # Test downloading with a bat pattern
    >>> with ub.CaptureStdout(suppress=False) as cap:
    >>>     download_helm_results.main(download_dir=dpath, runs='bad-pattern')
    >>> assert len(list(dpath.glob('**'))) == 0, 'should not have downloaded anything'
    >>> #
    >>> # Test downloading with a bat pattern
    >>> with ub.CaptureStdout(suppress=False) as cap:
    >>>     download_helm_results.main(download_dir=dpath, runs='med_qa:model=deepseek-ai_deepseek-v3', version='v1.13.0')
    >>> assert len(list(dpath.glob('**'))) > 200, 'should have only downloaded a few results'
"""
import re
import shutil
import sys
import ubelt as ub
import scriptconfig as scfg
from typing import List


class DownloadHelmConfig(scfg.DataConfig):
    """
    Download HELM benchmark run artifacts from the public GCS bucket.
    """
    __epilog__ = """
    Usage:
      ./download_helm_results.py <download_dir> [version]
      ./download_helm_results.py dir=<download_dir> [version=auto] [benchmark=lite]
      ./download_helm_results.py --list-benchmarks
      ./download_helm_results.py --list-versions [--benchmark=lite]

    Examples:
      ./download_helm_results.py /data/crfm-helm-public
      ./download_helm_results.py /data/crfm-helm-public --benchmark=helm
      ./download_helm_results.py /data/crfm-helm-public --benchmark=lite --version=v1.9.0
      ./download_helm_results.py dir=./data version=auto benchmark=lite

    Notes:
      - Requires: gsutil (Google Cloud SDK)
      - See [1]_ for official instructions
      - See [2]_ for available precomputed results

    References:
        .. [1] https://crfm-helm.readthedocs.io/en/latest/downloading_raw_results/
        .. [2] https://console.cloud.google.com/storage/browser/crfm-helm-public
    """
    download_dir = scfg.Value('', alias=['dir'], position=1, help='Destination directory')
    version = scfg.Value('auto', position=2, help='Optional version (e.g. v1.9.0)')
    benchmark = scfg.Value('lite', help='Benchmark name (e.g., lite, helm)')
    checksum = scfg.Value(False, isflag=True, help='Enable checksum-based comparison')
    install = scfg.Value(False, isflag=True, help='Auto-install gsutil on Debian/Ubuntu')

    list_benchmarks = scfg.Value(False, isflag=True, help='List available benchmarks and exit')
    list_versions = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            List available versions for the benchmark and exit
            '''))

    verbose = scfg.Value(False, isflag=True, help='Verbose output')
    bucket = scfg.Value("gs://crfm-helm-public", help="The storage bucket to download from.")

    runs = scfg.Value(None, help=ub.paragraph(
         '''
         Optional glob pattern to match specific run IDs within the chosen
         version.  Example: runs="*gpt4*", runs="llama-3-70b,claude-*"
         '''))  # empty means "download all runs in the version"


class ExitError(RuntimeError):
    def __init__(self, msg, code):
        super().__init__(msg, code)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def is_google_gsutil(gsutil_cmd: str, verbose: bool = False) -> bool:
    try:
        cp = ub.cmd([gsutil_cmd, "version"], verbose=verbose)
    except Exception:
        return False
    out = (cp.stdout or "") + (cp.stderr or "")
    return bool(re.search(r"^gsutil version:", out, flags=re.IGNORECASE | re.MULTILINE))


def apt_available() -> bool:
    return shutil.which("apt-get") is not None


def install_gsutil_ubuntu() -> None:
    eprint("Installing Google Cloud SDK (gsutil) via apt...")
    cmds = [
        ["sudo", "apt-get", "update", "-y"],
        ["sudo", "apt-get", "install", "-y", "apt-transport-https", "ca-certificates", "gnupg", "curl"],
        ["bash", "-lc",
         r"""curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
| sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg"""],
        ["bash", "-lc",
         r"""echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
| sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null"""],
        ["sudo", "apt-get", "update", "-y"],
        ["sudo", "apt-get", "install", "-y", "google-cloud-cli"],
    ]
    for c in cmds:
        ub.cmd(c, verbose=3)


def ensure_gsutil(install: bool) -> str:
    gsutil = shutil.which("gsutil")
    if gsutil and is_google_gsutil(gsutil):
        return gsutil

    eprint("Google Cloud 'gsutil' not found (or a conflicting 'gsutil' is first on PATH).")

    if install:
        if apt_available():
            install_gsutil_ubuntu()
        else:
            raise ExitError(code=1, msg=ub.paragraph(
                """
                Automatic install is only implemented for Debian/Ubuntu (apt).
                Install instructions: https://cloud.google.com/sdk/docs/install
                """))
    else:
        if sys.stdin.isatty() and apt_available():
            from rich import prompt
            ans = prompt.Confirm.ask("Install gsutil now via apt on Debian/Ubuntu?")
            if ans:
                install_gsutil_ubuntu()
            else:
                raise ExitError(code=1, msg=ub.paragraph(
                    """
                    Please install Google Cloud SDK and retry:
                    https://cloud.google.com/sdk/docs/install
                    """))
        else:
            raise ExitError(code=1, msg=ub.paragraph(
                """
                Please install Google Cloud SDK and retry:
                https://cloud.google.com/sdk/docs/install
                """))

    gsutil = shutil.which("gsutil")
    if not (gsutil and is_google_gsutil(gsutil)):
        raise ExitError(code=1, msg=ub.paragraph(
            """
            Error: gsutil still not available or not the Google Cloud version
            """))
    return gsutil


def version_key(v: str):
    """
    Turn strings like 'v1.9.0' or '1.9.0' into a comparable tuple (1,9,0,...).
    Non-numeric parts become zeros at the end to keep ordering stable.
    """
    v = v.strip().rstrip("/")
    v = v[1:] if v.lower().startswith("v") else v
    parts = re.split(r"[^\d]+", v)
    nums = []
    for p in parts:
        if p.isdigit():
            nums.append(int(p))
    return tuple(nums or [0])


def do_list_benchmarks(gsutil: str, bucket: str, verbose: bool = False) -> List[str]:
    cp = ub.cmd([gsutil, "ls", f"{bucket}/"], verbose=verbose)
    lines = [x.strip() for x in (cp.stdout or "").splitlines()]
    out = []
    for line in lines:
        m = re.match(rf"{re.escape(bucket)}/([^/]+)/?$", line)
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def do_list_versions(gsutil: str, bucket: str, bench: str, verbose: bool = False) -> List[str]:
    runs_path = f"{bucket}/{bench}/benchmark_output/runs"
    cp = ub.cmd([gsutil, "ls", f"{runs_path}/"], verbose=verbose)
    lines = [x.strip() for x in (cp.stdout or "").splitlines()]
    vers = []
    for line in lines:
        m = re.match(rf"{re.escape(runs_path)}/([^/]+)/?$", line)
        if m:
            vers.append(m.group(1))
    return sorted(set(vers), key=version_key)


def do_list_runs(gsutil: str, src_version_path: str, verbose: bool = False) -> List[str]:
    """
    List immediate child run directories under a version path:
      gs://.../<benchmark>/benchmark_output/runs/<version>/<run_id>/
    Returns a list of run_id strings (without trailing slash).
    """
    cp = ub.cmd([gsutil, "ls", f"{src_version_path}/"], verbose=verbose)
    lines = [x.strip() for x in (cp.stdout or "").splitlines()]
    out = []
    # Match: <src_version_path>/<run_id>/
    pat = re.compile(rf"{re.escape(src_version_path)}/([^/]+)/?$")
    for line in lines:
        m = pat.match(line)
        if m:
            out.append(m.group(1))
    # Some buckets may list files too; keep only unique run-like prefixes.
    return sorted(set(out))


def latest_version(gsutil: str, bucket: str, bench: str, verbose: bool = False) -> str:
    vers = do_list_versions(gsutil, bucket, bench, verbose=verbose)
    return vers[-1] if vers else ""


def gsutil_rsync(gsutil: str, src: str, dest: str, checksum: bool) -> None:
    cmd = [gsutil, "-m", "rsync", "-r"]
    if checksum:
        cmd.append("-c")
    cmd += [src, dest]
    ub.cmd(cmd, verbose=1, capture=False)


def do_requested_download(gsutil, src, dest, verbose, args):
    """
    Main download logic, either filtered or not.
    """
    import subprocess
    try:
        if args.runs:
            import kwutil
            pattern = kwutil.MultiPattern.coerce(args.runs)
            # Filter to a subset of run IDs by regex (comma-separated supported).
            all_runs = do_list_runs(gsutil, src, verbose=verbose)
            if not all_runs:
                eprint(f"No runs found under version path: {src}")
                return 1

            matched = [r for r in all_runs if pattern.match(r)]
            if not matched:
                eprint(f"No runs matched patterns {pattern} under {src}")
                eprint("Available runs:")
                for r in all_runs:
                    eprint(f"  - {r}")
                eprint(f"No runs matched patterns {pattern} under {src}. Choose a pattern matching some of the above")
                return 1

            print(f"Matching runs ({len(matched)}):")
            for r in matched:
                print(f"  - {r}")

            # Sync each selected run subdirectory independently.
            dest.mkdir(parents=True, exist_ok=True)
            for r in matched:
                sub_src = f"{src}/{r}"
                sub_dest = str(dest / r)
                ub.Path(sub_dest).mkdir(parents=True, exist_ok=True)
                gsutil_rsync(gsutil, sub_src, sub_dest, checksum=bool(args.checksum))
        else:
            # Download entire version as before.
            dest.mkdir(parents=True, exist_ok=True)
            gsutil_rsync(gsutil, src, str(dest), checksum=bool(args.checksum))

    except subprocess.CalledProcessError as ex:
        eprint("gsutil rsync failed.")
        if ex.stderr:
            eprint(ex.stderr.strip())
        return ex.returncode or 1
    print(f"Done. Files are under: {dest}")
    return 0


def main(argv=None, **kwargs) -> int:
    args = DownloadHelmConfig.cli(
        argv=argv, data=kwargs, strict=True, verbose='auto')
    verbose = bool(args.verbose)

    # Listing modes (no dir required)
    if args.list_benchmarks or args.list_versions:
        try:
            gsutil = ensure_gsutil(install=args.install)
        except ExitError as ex:
            eprint(ex.msg)
            return ex.code

        if args.list_benchmarks:
            benchmark_names = do_list_benchmarks(gsutil, args.bucket, verbose=verbose)
            for name in benchmark_names:
                print(name)
            return 0
        if args.list_versions:
            versions = do_list_versions(gsutil, args.bucket, args.benchmark, verbose=verbose)
            for v in versions:
                print(v)
            return 0

    # Require a destination directory for sync
    if not args.download_dir:
        eprint("Error: download directory not provided. Run with --help for usage")
        return 2

    gsutil = ensure_gsutil(install=args.install)

    # Resolve version if auto
    version = args.version
    if version == "auto":
        eprint(f"Resolving latest version for benchmark '{args.benchmark}'...")
        version = latest_version(gsutil, args.bucket, args.benchmark, verbose=verbose)
        if not version:
            eprint("Error: could not determine latest version (no runs found?).")
            return 1
        eprint(f"Using latest version: {version}")

    bucket_base = f"{args.bucket}/{args.benchmark}/benchmark_output/runs"
    src = f"{bucket_base}/{version}"
    dest_root = ub.Path(args.download_dir).expanduser().resolve() / args.benchmark / "benchmark_output" / "runs"
    dest = dest_root / version

    print(f"HELM benchmark: {args.benchmark}")
    print(f"Version:        {version}")
    print(f"Source:         {src}")
    print(f"Destination:    {dest}")
    print()

    # Idempotent sync
    ret = do_requested_download(gsutil, src, dest, verbose, args)
    return ret


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
