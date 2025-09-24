#!/usr/bin/env python3
"""
Download HELM benchmark run artifacts from the public GCS bucket.

- Will auto-detect the latest version if unspecified.
- Can choose a specific benchmark suite name, and a pattern to match only relevant results.
- Use --list-version and --list-benchmarks to explore available data

Example:
    >>> # xdoctest: +REQUIRES(module:gcsfs)
    >>> from magnet.backends.helm import download_helm_results
    >>> import ubelt as ub
    >>> #
    >>> # Test listing benchamrks
    >>> with ub.CaptureStdout(suppress=False) as cap:
    >>>     download_helm_results.main(argv=False, list_benchmarks=True)
    >>> assert len(cap.text.split()) >= 29
    >>> #
    >>> # Test listing versions
    >>> with ub.CaptureStdout(suppress=False) as cap:
    >>>     download_helm_results.main(argv=False, list_versions=True, benchmark='lite')
    >>> assert len(cap.text.split()) >= 14

Example:
    >>> # xdoctest: +REQUIRES(module:gcsfs)
    >>> from magnet.backends.helm import download_helm_results
    >>> import ubelt as ub
    >>> # Start fresh
    >>> dpath = ub.Path.appdir('magnet/tests/download_helm_list')
    >>> dpath.delete()
    >>> existing = [r / f for r, ds, fs in dpath.walk() for f in fs + ['.']]
    >>> assert len(existing) == 0, 'delete should remove everything'
    >>> #
    >>> # Test downloading with a bat pattern
    >>> with ub.CaptureStdout(suppress=False) as cap:
    >>>     download_helm_results.main(argv=False, download_dir=dpath, runs='bad-pattern')
    >>> existing = [r / f for r, ds, fs in dpath.walk() for f in fs + ['.']]
    >>> assert len(existing) == 0, 'should not have downloaded anything'
    >>> #
    >>> # Test downloading with a bat pattern
    >>> with ub.CaptureStdout(suppress=False) as cap:
    >>>     download_helm_results.main(argv=False, download_dir=dpath, runs='med_qa:model=deepseek-ai_deepseek-v3', version='v1.13.0')
    >>> existing = [r / f for r, ds, fs in dpath.walk() for f in fs + ['.']]
    >>> print(f'existing = {ub.urepr(existing, nl=1)}')
    >>> assert len(existing) == 14, 'should have only downloaded a few results'
"""

import re
import shutil
import sys
import ubelt as ub
import scriptconfig as scfg
from functools import cached_property
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
      # Show docs
      python -m magnet.backends.helm.download_helm_results --help

      # Explore
      python -m magnet.backends.helm.download_helm_results --list-benchmarks
      python -m magnet.backends.helm.download_helm_results --benchmark=lite --list-versions
      python -m magnet.backends.helm.download_helm_results --benchmark=lite --version=v1.9.0 --list-runs
      python -m magnet.backends.helm.download_helm_results --benchmark=lite --version=v1.9.0 --list-runs --runs "regex:.*subject=abstract.*model=.*llama.*"
      python -m magnet.backends.helm.download_helm_results --benchmark=lite --version=v1.9.0 --list-runs --runs "
          - wmt_14:language_pair=cs-en,model=meta_llama-*-vision*
          - narrative_qa:model=meta_llama-*-vision-instruct-turbo*
      "

      # Download
      python -m magnet.backends.helm.download_helm_results /data/crfm-helm-public
      python -m magnet.backends.helm.download_helm_results /data/crfm-helm-public --benchmark=ewok
      python -m magnet.backends.helm.download_helm_results /data/crfm-helm-public --benchmark=lite --version=v1.9.0

      #
      python -m magnet.backends.helm.download_helm_results /data/crfm-helm-public --benchmark=lite --version=v1.9.0 --runs regex:math:subject=precalculus,.*istruct-turbo

      python -m magnet.backends.helm.download_helm_results --dir=./data --version=auto --benchmark=lite

    Notes:
      - Requires: fsspec or gsutil (Google Cloud SDK)
      - See [1]_ for official instructions
      - See [2]_ for available precomputed results

    References:
        .. [1] https://crfm-helm.readthedocs.io/en/latest/downloading_raw_results/
        .. [2] https://console.cloud.google.com/storage/browser/crfm-helm-public
    """
    download_dir = scfg.Value(
        '', alias=['dir'], position=1, help='Destination directory'
    )
    benchmark = scfg.Value('lite', position=2, help='Benchmark name (e.g., lite, helm)')
    version = scfg.Value(
        'auto',
        position=3,
        help='Benchmark version (e.g. v1.9.0). If "auto", will default to latest',
    )

    runs = scfg.Value(
        None,
        type=str,
        help=ub.paragraph(
            """
            Optional glob pattern (or kwutil MultiPattern) to match specific
            run IDs within the chosen version.  E.g.: runs="*gpt4*",
            runs="regex:llama-3-70b,claude-.*", or ['patternA', 'patternB']
            """
        ),
    )  # empty means "download all runs in the version"

    list_benchmarks = scfg.Value(
        False,
        isflag=True,
        group='listers',
        help='List available benchmarks and exit',
    )
    list_versions = scfg.Value(
        False,
        isflag=True,
        group='listers',
        help=ub.paragraph(
            """
            List available versions for the benchmark and exit
            """
        ),
    )
    list_runs = scfg.Value(
        False,
        isflag=True,
        group='listers',
        help=ub.paragraph(
            """
            List available runs for the benchmark / version and then exit
            """
        ),
    )

    verbose = scfg.Value(False, isflag=True, help='Verbose output', group='logging')
    bucket = scfg.Value(
        'gs://crfm-helm-public',
        help='The storage bucket to download from. No need to change this.',
        group='behavior',
    )
    checksum = scfg.Value(
        False, isflag=True, help='Enable checksum-based comparison', group='behavior'
    )
    backend = scfg.Value(
        'fsspec',
        choices=['gsutil', 'fsspec'],
        group='behavior',
        help=ub.paragraph(
            """
            Choose transfer/listing backend: "gsutil" (CLI) or "fsspec" (pure
            Python via gcsfs).
            """
        ),
    )
    install = scfg.Value(
        False,
        isflag=True,
        group='behavior',
        help='Auto-install gsutil on Debian/Ubuntu. Only relevant for gsutil backend',
    )


class ExitError(RuntimeError):
    def __init__(self, msg, code):
        super().__init__(msg, code)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# ===============================
# Backend abstractions
# ===============================


class _GSBaseBackend:
    """
    Unsure if the cli or fsspec is more reliable, for now impelemnt both as a
    backend and abstract over them.

    Minimal interface the main flow depends on.
      - list_benchmarks() -> List[str]
      - list_versions(bench) -> List[str]
      - latest_version(bench) -> str
      - list_runs(bench, version) -> List[str]
      - download_version(bench, version, dest, checksum=False) -> None
      - download_runs(bench, version, dest, run_ids, checksum=False) -> None

    Example:
        >>> # xdoctest: +REQUIRES(module:gcsfs)
        >>> # Test backends are the same
        >>> from magnet.backends.helm.download_helm_results import *  # NOQA
        >>> import pytest
        >>> if not GSCLIBackend.is_available():
        >>>     pytest.skip('cli tool not available, cannot test')
        >>> bucket = "gs://crfm-helm-public"
        >>> benchmark = 'lite'
        >>> version = 'v1.0.0'
        >>> backend1 = GSCLIBackend(bucket)
        >>> backend2 = GSFSSspecBackend(bucket)
        >>> result1 = backend1.list_benchmarks()
        >>> result2 = backend2.list_benchmarks()
        >>> assert result1 == result2
        >>> result1 = backend1.list_versions(benchmark)
        >>> result2 = backend2.list_versions(benchmark)
        >>> assert result1 == result2
        >>> result1 = backend1.list_runs(benchmark, version)
        >>> result2 = backend2.list_runs(benchmark, version)
        >>> assert result1 == result2
        >>> run_ids = ['gsm:model=meta_llama-2-13b']
        >>> dpath1 = ub.Path.appdir('magnet/tests/gsbackends/cli').delete()
        >>> dpath2 = ub.Path.appdir('magnet/tests/gsbackends/fsspec').delete()
        >>> with ub.Timer(f'backend: {backend1}'):
        >>>     backend1.download_runs(benchmark, version, dpath1, run_ids)
        >>> with ub.Timer(f'backend: {backend2}'):
        >>>     backend2.download_runs(benchmark, version, dpath2, run_ids)
        >>> result1 = sorted([p.relative_to(dpath1) for p in dpath1.ls('**')])
        >>> result2 = sorted([p.relative_to(dpath2) for p in dpath2.ls('**')])
        >>> assert result2 == result1
    """

    def __init__(self, bucket: str):
        self.bucket = bucket.rstrip('/')

    # Optional helper for uniform GS path handling
    def _version_relpath(self, version_src_gs: str) -> str:
        # Convert 'gs://bucket/x/y' -> 'x/y'
        return version_src_gs.replace(self.bucket + '/', '', 1).lstrip('/')


class GSCLIBackend(_GSBaseBackend):
    """
    gsutil/CLI implementation.

    Note: this backend can likely be removed if we find that fsspec doesn't
    have any issues, so far it seems faster, better, and more reliable than
    using the cli tool. Leaving this in for now.
    """

    def __init__(self, bucket: str):
        super().__init__(bucket)

    @cached_property
    def gsutil(self):
        return self.__class__.ensure_gsutil()

    @classmethod
    def is_available(cls):
        return cls._find_gsutil() is not None

    @classmethod
    def _find_gsutil(cls):
        gsutil = shutil.which('gsutil')
        if gsutil and cls._is_google_gsutil(gsutil):
            return gsutil

    @classmethod
    def ensure_gsutil(cls, install: bool = False) -> str:
        gsutil = cls._find_gsutil()
        if gsutil:
            return gsutil

        eprint(
            "Google Cloud 'gsutil' not found (or a conflicting 'gsutil' is first on PATH)."
        )

        if install:
            if cls._apt_available():
                cls._install_gsutil_ubuntu()
            else:
                raise ExitError(
                    code=1,
                    msg=ub.paragraph(
                        """
                        Automatic install is only implemented for Debian/Ubuntu (apt).
                        Install instructions: https://cloud.google.com/sdk/docs/install
                        """
                    ),
                )
        else:
            if sys.stdin.isatty() and cls._apt_available():
                from rich import prompt

                ans = prompt.Confirm.ask('Install gsutil now via apt on Debian/Ubuntu?')
                if ans:
                    cls._install_gsutil_ubuntu()
                else:
                    raise ExitError(
                        code=1,
                        msg=ub.paragraph(
                            """
                            Please install Google Cloud SDK and retry:
                            https://cloud.google.com/sdk/docs/install
                            """
                        ),
                    )
            else:
                raise ExitError(
                    code=1,
                    msg=ub.paragraph(
                        """
                        Please install Google Cloud SDK and retry:
                        https://cloud.google.com/sdk/docs/install
                        """
                    ),
                )

        gsutil = shutil.which('gsutil')
        if not gsutil:
            raise ExitError(
                code=1,
                msg=ub.paragraph(
                    """
                    Error: gsutil still not available.
                    """
                ),
            )
            if cls._is_google_gsutil(gsutil):
                raise ExitError(
                    code=1,
                    msg=ub.paragraph(
                        """
                        Error: gsutil exists, but is not the Google Cloud version.
                        """
                    ),
                )
        return gsutil

    @classmethod
    def _is_google_gsutil(cls, gsutil_cmd: str, verbose: bool = False) -> bool:
        try:
            cp = ub.cmd([gsutil_cmd, 'version'], verbose=verbose)
        except Exception:
            return False
        out = (cp.stdout or '') + (cp.stderr or '')
        return bool(
            re.search(r'^gsutil version:', out, flags=re.IGNORECASE | re.MULTILINE)
        )

    @classmethod
    def _apt_available(cls) -> bool:
        return shutil.which('apt-get') is not None

    @classmethod
    def _install_gsutil_ubuntu(cls) -> None:
        eprint('Installing Google Cloud SDK (gsutil) via apt...')
        cmds = [
            ['sudo', 'apt-get', 'update', '-y'],
            [
                'sudo',
                'apt-get',
                'install',
                '-y',
                'apt-transport-https',
                'ca-certificates',
                'gnupg',
                'curl',
            ],
            [
                'bash',
                '-lc',
                r"""curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg""",
            ],
            [
                'bash',
                '-lc',
                r"""echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null""",
            ],
            ['sudo', 'apt-get', 'update', '-y'],
            ['sudo', 'apt-get', 'install', '-y', 'google-cloud-cli'],
        ]
        for c in cmds:
            ub.cmd(c, verbose=3)

    def list_benchmarks(self, verbose: bool = False) -> List[str]:
        cp = ub.cmd([self.gsutil, 'ls', f'{self.bucket}/'], verbose=verbose)
        lines = [x.strip() for x in (cp.stdout or '').splitlines()]
        out = []
        for line in lines:
            m = re.match(rf'{re.escape(self.bucket)}/([^/]+)/?$', line)
            if m:
                out.append(m.group(1))
        return sorted(set(out))

    def list_versions(self, bench: str, verbose: bool = False) -> List[str]:
        runs_path = f'{self.bucket}/{bench}/benchmark_output/runs'
        cp = ub.cmd([self.gsutil, 'ls', f'{runs_path}/'], verbose=verbose)
        lines = [x.strip() for x in (cp.stdout or '').splitlines()]
        vers = []
        for line in lines:
            m = re.match(rf'{re.escape(runs_path)}/([^/]+)/?$', line)
            if m:
                vers.append(m.group(1))
        return sorted(set(vers), key=_version_key)

    def latest_version(self, bench: str, verbose: bool = False) -> str:
        vers = self.list_versions(bench, verbose=verbose)
        return vers[-1] if vers else ''

    def list_runs(self, bench: str, version: str, verbose: bool = False) -> List[str]:
        """
        List runs under a specific benchmark and version
        Returns a list of run_id strings (without trailing slash).
        """
        bucket_base = f'{self.bucket}/{bench}/benchmark_output/runs'
        version_src_gs = f'{bucket_base}/{version}'
        cp = ub.cmd([self.gsutil, 'ls', f'{version_src_gs}/'], verbose=verbose)
        lines = [x.strip() for x in (cp.stdout or '').splitlines()]
        out = []
        # Match: <src_version_path>/<run_id>/
        pat = re.compile(rf'{re.escape(version_src_gs)}/([^/]+)/?$')
        for line in lines:
            m = pat.match(line)
            if m:
                out.append(m.group(1))
        # Some buckets may list files too; keep only unique run-like prefixes.
        return sorted(set(out))

    # Transfers
    def download_version(
        self, bench: str, version: str, dest: ub.Path, checksum: bool = False
    ) -> None:
        bucket_base = f'{self.bucket}/{bench}/benchmark_output/runs'
        version_src_gs = f'{bucket_base}/{version}'
        dest = ub.Path(dest).ensuredir()
        self._gsutil_rsync(version_src_gs, str(dest), checksum=checksum)

    def download_runs(
        self,
        bench: str,
        version: str,
        dest: ub.Path,
        run_ids: List[str],
        checksum: bool = False,
    ) -> None:
        bucket_base = f'{self.bucket}/{bench}/benchmark_output/runs'
        version_src_gs = f'{bucket_base}/{version}'
        dest = ub.Path(dest).ensuredir()
        for r in run_ids:
            sub_src = f'{version_src_gs}/{r}'
            sub_dest = dest / r
            sub_dest.mkdir(parents=True, exist_ok=True)
            self._gsutil_rsync(sub_src, str(sub_dest), checksum=checksum)

    def _gsutil_rsync(self, src: str, dest: str, checksum: bool) -> None:
        cmd = [self.gsutil, '-m', 'rsync', '-r']
        if checksum:
            cmd.append('-c')
        cmd += [src, dest]
        ub.cmd(cmd, verbose=1, capture=False)


class GSFSSspecBackend(_GSBaseBackend):
    """
    Pure-Python fsspec/gcsfs implementation
    (anonymous access for public buckets).
    """

    def __init__(self, bucket: str):
        super().__init__(bucket)
        try:
            import fsspec  # noqa: F401
        except Exception as ex:
            raise ExitError(
                msg=ub.paragraph(
                    f"""
                    backend=gcsfs requested, but fsspec/gcsfs is not installed.
                    Please: pip install gcsfs fsspec
                    (original error: {ex})
                    """
                ),
                code=1,
            )
        self.fs = fsspec.filesystem('gcs', token='anon')

    # Listing
    def list_benchmarks(self, verbose: bool = False) -> List[str]:
        root = _strip_gs(self.bucket).rstrip('/')
        try:
            entries = self.fs.ls(root + '/', detail=True)
        except FileNotFoundError:
            return []
        out = []
        for e in entries:
            if e.get('type') == 'directory':
                out.append(e['name'].rstrip('/').split('/')[-1])
        return sorted(set(out))

    def list_versions(self, bench: str, verbose: bool = False) -> List[str]:
        base = _strip_gs(f'{self.bucket}/{bench}/benchmark_output/runs').rstrip('/')
        try:
            entries = self.fs.ls(base + '/', detail=True)
        except FileNotFoundError:
            return []
        vers = []
        for e in entries:
            if e.get('type') == 'directory':
                vers.append(e['name'].rstrip('/').split('/')[-1])
        return sorted(set(vers), key=_version_key)

    def latest_version(self, bench: str, verbose: bool = False) -> str:
        vers = self.list_versions(bench, verbose=verbose)
        return vers[-1] if vers else ''

    def list_runs(self, bench: str, version: str, verbose: bool = False) -> List[str]:
        bucket_base = f'{self.bucket}/{bench}/benchmark_output/runs'
        version_src_gs = f'{bucket_base}/{version}'
        base = _strip_gs(version_src_gs).rstrip('/')
        try:
            entries = self.fs.ls(base + '/', detail=True)
        except FileNotFoundError:
            return []
        runs = []
        for e in entries:
            if e.get('type') == 'directory':
                runs.append(e['name'].rstrip('/').split('/')[-1])
        return sorted(set(runs))

    # Transfers
    def download_version(
        self, bench: str, version: str, dest: ub.Path, checksum: bool = False
    ) -> None:
        bucket_base = f'{self.bucket}/{bench}/benchmark_output/runs'
        version_src_gs = f'{bucket_base}/{version}'
        if checksum:
            eprint(
                'Note: checksum is not supported with backend=gcsfs; proceeding without.'
            )
        src = _strip_gs(version_src_gs).rstrip('/')
        dest = ub.Path(dest).ensuredir()
        self.fs.get(src, str(dest), recursive=True)

    def download_runs(
        self,
        bench: str,
        version: str,
        dest: ub.Path,
        run_ids: List[str],
        checksum: bool = False,
    ) -> None:
        bucket_base = f'{self.bucket}/{bench}/benchmark_output/runs'
        version_src_gs = f'{bucket_base}/{version}'
        if checksum:
            eprint(
                'Note: checksum is not supported with backend=gcsfs; proceeding without.'
            )
        base = _strip_gs(version_src_gs).rstrip('/')
        dest = ub.Path(dest).ensuredir()
        for r in run_ids:
            self.fs.get(f'{base}/{r}', str(dest / r), recursive=True)


def _strip_gs(url: str) -> str:
    return url.replace('gs://', '', 1) if url.startswith('gs://') else url


def _version_key(v: str):
    """
    Turn strings like 'v1.9.0' or '1.9.0' into a comparable tuple (1,9,0,...).
    Non-numeric parts become zeros at the end to keep ordering stable.
    """
    v = v.strip().rstrip('/')
    v = v[1:] if v.lower().startswith('v') else v
    parts = re.split(r'[^\d]+', v)
    nums = []
    for p in parts:
        if p.isdigit():
            nums.append(int(p))
    return tuple(nums or [0])


def filter_runs(all_runs, runs):
    import kwutil

    pattern = kwutil.MultiPattern.coerce(runs)
    matched = [r for r in all_runs if pattern.match(r)]
    return matched


def _do_requested_download(backend, benchmark, version, dest, verbose, runs, checksum):
    """
    Main download logic, either filtered or not.
    """
    bucket_base = f'{backend.bucket}/{benchmark}/benchmark_output/runs'
    src = f'{bucket_base}/{version}'

    import subprocess

    try:
        if runs:
            import kwutil

            # Filter to a subset of run IDs by regex (comma-separated supported).
            all_runs = backend.list_runs(benchmark, version, verbose=verbose)
            if not all_runs:
                eprint(f'No runs found under version path: {src}')
                return 1

            pattern = kwutil.MultiPattern.coerce(runs)
            matched = filter_runs(all_runs, pattern)
            if not matched:
                eprint(f'No runs matched patterns {pattern} under {src}')
                eprint('Available runs:')
                for r in all_runs:
                    eprint(f'  - {r}')
                eprint(
                    f'No runs matched patterns {pattern} under {src}. Choose a pattern matching some of the above'
                )
                return 1

            print(f'Matching runs ({len(matched)}):')
            for r in matched:
                print(f'  - {r}')

            # Sync each selected run subdirectory independently.
            dest.mkdir(parents=True, exist_ok=True)
            backend.download_runs(
                benchmark, version, dest, matched, checksum=bool(checksum)
            )
        else:
            # Download entire version.
            backend.download_version(benchmark, version, dest, checksum=bool(checksum))

    except subprocess.CalledProcessError as ex:
        eprint('gsutil rsync failed.')
        if ex.stderr:
            eprint(ex.stderr.strip())
        return ex.returncode or 1
    print(f'Done. Files are under: {dest}')
    return 0


def main(argv=None, **kwargs) -> int:
    args = DownloadHelmConfig.cli(
        argv=argv, data=kwargs, strict=True, verbose='auto', special_options=False
    )
    verbose = bool(args.verbose)

    import kwutil

    benchmark = args.benchmark
    runs = kwutil.Yaml.coerce(args.runs)
    checksum = args.checksum

    # Choose backend for list operations
    try:
        if args.backend == 'gsutil':
            backend = GSCLIBackend(args.bucket)
            backend.ensure_gsutil(install=args.install)
        else:
            backend = GSFSSspecBackend(args.bucket)
    except ExitError as ex:
        eprint(ex.msg)
        return ex.code

    if args.list_benchmarks:
        for name in backend.list_benchmarks(verbose=verbose):
            print(name)
        return 0
    if args.list_versions:
        for v in backend.list_versions(benchmark, verbose=verbose):
            print(v)
        return 0

    # Resolve version if auto
    version = args.version
    if version == 'auto':
        eprint(
            f"Resolving latest version for benchmark '{args.benchmark}' (backend={args.backend})..."
        )
        version = backend.latest_version(benchmark, verbose=verbose)
        if not version:
            eprint('Error: could not determine latest version (no runs found?).')
            return 1
        eprint(f'Using latest version: {version}')

    if args.list_runs:
        all_runs = backend.list_runs(benchmark, version, verbose=verbose)
        if runs:
            matched = filter_runs(all_runs, runs)
        else:
            matched = all_runs
        for v in matched:
            print(v)
        return 0

    # Require a destination directory for sync
    if not args.download_dir:
        eprint('Error: download directory not provided. Run with --help for usage')
        return 2

    # TODO: probably should have the backend class handle this path stuff to
    # keep the API at the level of benchmark (i.e. suite) name, versions, and
    # run names.
    bucket_base = f'{backend.bucket}/{benchmark}/benchmark_output/runs'
    src = f'{bucket_base}/{version}'
    download_dir = ub.Path(args.download_dir)
    dest_root = download_dir / benchmark / 'benchmark_output' / 'runs'
    dest = dest_root / version

    print(f'HELM benchmark: {args.benchmark}')
    print(f'Version:        {version}')
    print(f'Source:         {src}')
    print(f'Destination:    {dest}')
    print()

    # Idempotent sync
    ret = _do_requested_download(
        backend, benchmark, version, dest, verbose, runs, checksum
    )
    return ret


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
