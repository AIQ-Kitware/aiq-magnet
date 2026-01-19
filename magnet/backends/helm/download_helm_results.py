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
    >>> assert len(cap.text.split()) >= 24
    >>> #
    >>> # Test listing versions
    >>> with ub.CaptureStdout(suppress=False) as cap:
    >>>     download_helm_results.main(argv=False, list_versions=True, benchmark='lite')
    >>> assert len(cap.text.split()) >= 14

    >>> # Test listing runs (using classic benchmark, which tests a special case)
    >>> with ub.CaptureStdout(suppress=False) as cap:
    >>>     download_helm_results.main(argv=False, list_runs=True, version='v0.4.0', benchmark='classic')
    >>> assert len(cap.text.split()) >= 70

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
    # hack, scriptconfig should allow modals to overwrite this in the context
    # of usage, not definition in a future version, for now this does what we
    # want.
    # __command__ = 'helm'

    __epilog__ = """
    Usage:
      magnet download helm <download_dir> <benchmark_pattern> <version_pattern>
      magnet download helm --dir <download_dir> [--version <pattern|latest>] [--benchmark <pattern>] [--runs <pattern>]
      magnet download helm --list-benchmarks
      magnet download helm --list-versions [--benchmark <pattern>]
      magnet download helm --list-runs [--benchmark <pattern>] [--version <pattern|latest>]

    Examples:
      # Show docs
      magnet download helm --help

      # Explore
      magnet download helm --list-benchmarks
      magnet download helm --benchmark=lite --list-versions
      magnet download helm --benchmark=lite --version=v1.9.0 --list-runs
      magnet download helm --benchmark=lite --version=v1.9.0 --list-runs --runs "regex:.*subject=abstract.*model=.*llama.*"
      magnet download helm --benchmark=lite --version=v1.9.0 --list-runs --runs "
          - wmt_14:language_pair=cs-en,model=meta_llama-*-vision*
          - narrative_qa:model=meta_llama-*-vision-instruct-turbo*
      "

      # Download (single benchmark/version)
      magnet download helm /data/crfm-helm-public
      magnet download helm /data/crfm-helm-public --benchmark=ewok
      magnet download helm /data/crfm-helm-public --benchmark=lite --version=v1.9.0
      magnet download helm /data/crfm-helm-public --benchmark=lite --version=v1.9.0 --runs "regex:math:subject=precalculus,.*istruct-turbo"

      # Download (multi-benchmark / multi-version via patterns)
      magnet download helm /data/crfm-helm-public --benchmark="lite|ewok" --version="v1.9.0|v1.10.0"
      magnet download helm /data/crfm-helm-public --benchmark="regex:.*" --version="regex:.*"  # everything

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
    benchmark = scfg.Value(
        'lite',
        position=2,
        help='Benchmark name (e.g., lite, helm, classic). Use a kwutil.MultiPattern for multi-select (e.g. "lite|ewok" or "regex:.*")',
    )
    version = scfg.Value(
        'latest',
        position=3,
        help='Benchmark version (e.g. v1.9.0). If latest/auto, uses most recent. You may also use a kwutil.MultiPattern to select multiple versions.',
    )
    stop_on_error = scfg.Value(
        False,
        isflag=True,
        group='behavior',
        help='When downloading multiple benchmarks/versions, stop on first error',
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
    def __init__(self, msg: str, code: int):
        super().__init__(msg, code)
    
    @property
    def msg(self) -> str:
        return self.args[0]
    
    @property
    def code(self) -> int:
        return self.args[1]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# ===============================
# Backend abstractions
# ===============================


class GsutilStorageBackend:
    """Implementation via Google Cloud SDK `gsutil` CLI.

    Note: this backend can likely be removed if we find that fsspec doesn't
    have any issues, so far it seems faster, better, and more reliable than
    using the cli tool. Leaving this in for now.
    """

    def __init__(self, bucket):
        self.bucket = bucket.rstrip('/')

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
        out = str(cp.stdout or '') + str(cp.stderr or '')
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

    # ---- protocol ----
    def list_dirs(self, prefix: str) -> List[str]:
        # Normalize to gs://...
        prefix = prefix.rstrip('/') + '/'
        cp = ub.cmd([self.gsutil, 'ls', prefix], verbose=0)
        lines = [x.strip() for x in str(cp.stdout or '').splitlines()]
        out = []
        # match 'gs://bucket/prefix/child/'
        pat = re.compile(rf'^{re.escape(prefix)}([^/]+)/?$')
        for line in lines:
            m = pat.match(line)
            if m:
                out.append(m.group(1))
        return sorted(set(out))

    def download_tree(
        self, src_prefix: str, dest_dir: ub.Path, checksum: bool = False
    ) -> None:
        dest_dir.ensuredir()
        cmd = [self.gsutil, '-m', 'rsync', '-r']
        if checksum:
            cmd.append('-c')
        cmd += [src_prefix, str(dest_dir)]
        ub.cmd(cmd, verbose=1, capture=False)


class FsspecStorageBackend:
    """Pure-Python implementation via fsspec/gcsfs (anonymous access)."""

    def __init__(self, bucket: str):
        self.bucket = bucket.rstrip('/')
        try:
            import fsspec  # type: ignore
        except Exception as ex:  # pragma: no cover (import-time edge)
            raise ExitError(
                f'backend=fsspec requested, but fsspec/gcsfs is not installed: {ex}', 1
            )
        self.fs = fsspec.filesystem('gcs', token='anon')

    def list_dirs(self, prefix: str) -> List[str]:
        """
        Ignore:
            self = FsspecStorageBackend('gs://crfm-helm-public')
            prefix = 'gs://crfm-helm-public/lite/benchmark_output/runs/v1.0.0'
            self.list_dirs(prefix)
        """
        # Accept either 'gs://...' or 'bucket/...'
        root = _strip_gs(prefix).rstrip('/') + '/'
        try:
            entries = self.fs.ls(root, detail=True)
        except FileNotFoundError:
            return []
        out = []
        for e in entries:
            if e.get('type') == 'directory':
                out.append(e['name'].rstrip('/').split('/')[-1])
        return sorted(set(out))

    def download_tree(
        self, src_prefix: str, dest_dir: ub.Path, checksum: bool = False
    ) -> None:
        if checksum:
            eprint(
                'Note: checksum verification is not supported with fsspec; proceeding without it.'
            )
        base = _strip_gs(src_prefix).rstrip('/')
        dest_dir.ensuredir()
        from fsspec.callbacks import TqdmCallback
        # TODO: can we use fsspec.generic.rsync here?
        callback = TqdmCallback(tqdm_kwargs={"desc": f"Downloading {base}"})
        self.fs.get(base, str(dest_dir.parent) + '/', recursive=True, callback=callback)


class HelmRemoteStore:
    """
    Using some abstract backend storage, provide a way to navivage and download
    precomptued HELM results.

    Example:
        >>> # xdoctest: +REQUIRES(module:gcsfs)
        >>> from magnet.backends.helm.download_helm_results import *  # NOQA
        >>> self = HelmRemoteStore()
        >>> benchmarks = self.list_benchmarks()
        >>> benchmark = benchmarks[0]
        >>> verions = self.list_versions(benchmark)
        >>> verion = verions[0]
        >>> runs = self.list_runs(benchmark, verion)
        >>> print(f'benchmarks = {ub.urepr(benchmarks, nl=0)}')
        >>> print(f'verions = {ub.urepr(verions, nl=0)}')
        >>> print(f'runs = {ub.urepr(runs, nl=0)}')

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> # xdoctest: +REQUIRES(module:gcsfs)
        >>> # Test backends are the same
        >>> from magnet.backends.helm.download_helm_results import *  # NOQA
        >>> import pytest
        >>> if not GsutilStorageBackend.is_available():
        >>>     pytest.skip('cli tool not available, cannot test')
        >>> benchmark = 'lite'
        >>> version = 'v1.0.0'
        >>> storage1 = HelmRemoteStore(backend='gsutil')
        >>> storage2 = HelmRemoteStore(backend='fsspec')
        >>> result1 = storage1.list_benchmarks()
        >>> result2 = storage2.list_benchmarks()
        >>> assert result1 == result2
        >>> result1 = storage1.list_versions(benchmark)
        >>> result2 = storage2.list_versions(benchmark)
        >>> assert result1 == result2
        >>> result1 = storage1.list_runs(benchmark, version)
        >>> result2 = storage2.list_runs(benchmark, version)
        >>> assert result1 == result2
        >>> run_ids = ['gsm:model=meta_llama-2-13b']
        >>> dpath1 = ub.Path.appdir('magnet/tests/gsbackends/cli').delete()
        >>> dpath2 = ub.Path.appdir('magnet/tests/gsbackends/fsspec').delete()
        >>> with ub.Timer(f'backend: {storage1}'):
        >>>     storage1.download_runs(benchmark, version, dpath1, run_ids)
        >>> with ub.Timer(f'backend: {storage2}'):
        >>>     storage2.download_runs(benchmark, version, dpath2, run_ids)
        >>> result1 = sorted([r.relative_to(dpath1) / f for r, ds, fs in dpath1.walk() for f in fs + ['.']])
        >>> result2 = sorted([r.relative_to(dpath2) / f for r, ds, fs in dpath2.walk() for f in fs + ['.']])
        >>> assert result2 == result1
    """
    def __init__(self, bucket='gs://crfm-helm-public', backend='fsspec'):
        if backend == 'fsspec':
            self.backend = FsspecStorageBackend(bucket=bucket)
        elif backend == 'gsutil':
            self.backend = GsutilStorageBackend(bucket=bucket)

    @property
    def bucket(self) -> str:
        return self.backend.bucket

    # --- path helpers ---
    def _runs_root(self, benchmark: str) -> str:
        # HELM layout quirk: classic lives at top-level benchmark_output
        if benchmark == 'classic':
            return f'{self.bucket}/benchmark_output/runs'
        return f'{self.bucket}/{benchmark}/benchmark_output/runs'

    # --- list API ---
    def list_benchmarks(self) -> List[str]:
        # everything at bucket root are candidate benchmarks; filter out non-bench dirs
        names = set(self.backend.list_dirs(self.bucket))
        # include classic; remove non-bench directories we know about
        names.add('classic')
        blocklist = {
            'benchmark_output',
            'assets',
            'tmp',
            'config',
            'prod_env',
            'source_datasets',
        }
        return sorted(names - blocklist)

    def list_versions(self, benchmark: str) -> List[str]:
        from packaging.version import parse as Version, InvalidVersion
        root = self._runs_root(benchmark)
        vers = self.backend.list_dirs(root)
        try:
            # try to use proper version parsing
            return sorted(set(vers), key=Version)
        except InvalidVersion:
            # fallback
            return sorted(set(vers), key=_version_key)

    def latest_version(self, benchmark: str) -> str:
        # NOTE: this doesn't always order non standard versions correctly
        # e.g. (v1.1.0-preview)
        vers = self.list_versions(benchmark)
        return vers[-1] if vers else ''

    def list_runs(self, benchmark: str, version: str) -> List[str]:
        root = self._runs_root(benchmark)
        return self.backend.list_dirs(f'{root}/{version}')

    # --- download API ---
    def download_version(
        self, benchmark: str, version: str, dest: ub.Path, *, checksum: bool = False
    ) -> None:
        root = self._runs_root(benchmark)
        self.backend.download_tree(f'{root}/{version}', dest, checksum=checksum)

    def download_runs(
        self,
        benchmark: str,
        version: str,
        dest: ub.Path,
        run_ids: List[str],
        *,
        checksum: bool = False,
    ) -> None:
        root = self._runs_root(benchmark)
        for run_id in run_ids:
            run_dpath = (dest / run_id).ensuredir()
            self.backend.download_tree(
                f'{root}/{version}/{run_id}', run_dpath, checksum=checksum
            )


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


# If a selector contains any characters outside of this set, treat it as a
# kwutil.MultiPattern (i.e. it can match multiple items).
_SIMPLE_SELECTOR_RE = re.compile(r'^[\w\-.]+$')


def _looks_like_single_selector(text: str) -> bool:
    """Heuristic: if it only contains identifier-ish characters, treat as single value.

    Any other characters (e.g. '*', ',', ':', '[', ']', whitespace) are
    interpreted as a MultiPattern expression, which may match multiple items.
    """
    return bool(_SIMPLE_SELECTOR_RE.match(text or ''))


def filter_runs(all_runs, runs):
    import kwutil

    pattern = kwutil.MultiPattern.coerce(runs)
    matched = [r for r in all_runs if pattern.match(r)]
    return matched


def _do_requested_download(storage, benchmark, version, dest, verbose, runs, checksum):
    """
    Main download logic, either filtered or not.
    """
    bucket_base = storage._runs_root(benchmark)
    src = f'{bucket_base}/{version}'

    import subprocess

    try:
        if runs:
            import kwutil

            # Filter to a subset of run IDs by regex (comma-separated supported).
            all_runs = storage.list_runs(benchmark, version)
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
            storage.download_runs(
                benchmark, version, dest, matched, checksum=bool(checksum)
            )
        else:
            # Download entire version.
            storage.download_version(benchmark, version, dest, checksum=bool(checksum))

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
    benchmark_arg = args.benchmark
    version_arg = args.version

    try:
        runs = kwutil.Yaml.coerce(args.runs, backend='pyyaml')
    except Exception:
        # Simple glob strings can be invalid yaml, so account for that.
        runs = args.runs
    checksum = args.checksum

    # Choose backend for list operations
    try:
        storage = HelmRemoteStore(args.bucket, backend=args.backend)
    except ExitError as ex:
        eprint(ex.msg)
        return ex.code

    # --- listing modes ---
    if args.list_benchmarks:
        for name in storage.list_benchmarks():
            print(name)
        return 0

    def resolve_benchmarks(selector: str) -> List[str]:
        """Resolve benchmark selector.

        - If the selector looks like a single identifier, we assume it refers
          to one benchmark and **avoid** listing benchmarks first.
        - Otherwise, treat it as a kwutil.MultiPattern and match against the
          available benchmarks.
        """
        selector = (selector or '').strip()
        if _looks_like_single_selector(selector):
            return [selector]
        import kwutil
        pat = kwutil.MultiPattern.coerce(selector)
        all_benchmarks = storage.list_benchmarks()
        return [b for b in all_benchmarks if pat.match(b)]

    def resolve_versions(benchmark: str, selector: str) -> List[str]:
        """Resolve version selector for a benchmark.

        - 'latest' / 'auto' resolves to the most recent version.
        - If it looks like a single identifier, treat as a single version.
        - Otherwise treat as kwutil.MultiPattern and match against versions.
        """
        selector = (selector or '').strip()
        if selector in {'latest', 'auto'}:
            v = storage.latest_version(benchmark)
            return [v] if v else []
        if _looks_like_single_selector(selector):
            return [selector]
        import kwutil
        pat = kwutil.MultiPattern.coerce(selector)
        all_versions = storage.list_versions(benchmark)
        return [v for v in all_versions if pat.match(v)]

    # Resolve benchmarks set
    benchmark_list = resolve_benchmarks(benchmark_arg)
    if not benchmark_list:
        eprint(f"No benchmarks matched selector '{benchmark_arg}'")
        return 1

    if args.list_versions:
        for benchmark in benchmark_list:
            for version in storage.list_versions(benchmark):
                # If listing many benchmarks, prefix to keep output unambiguous
                if len(benchmark_list) > 1:
                    print(f'{benchmark}	{version}')
                else:
                    print(version)
        return 0

    # list-runs mode
    if args.list_runs:
        # Resolve versions per benchmark. If selector is a MultiPattern, it may
        # match multiple versions.
        for benchmark in benchmark_list:
            version_list = resolve_versions(benchmark, version_arg)
            for version in version_list:
                all_runs = storage.list_runs(benchmark, version)
                if runs:
                    matched = filter_runs(all_runs, runs)
                else:
                    matched = all_runs
                for r in matched:
                    # Prefix with benchmark/version when output would otherwise be ambiguous.
                    if len(benchmark_list) > 1 or len(version_list) > 1:
                        print(f'{benchmark}\t{version}\t{r}')
                    else:
                        print(r)
        return 0

    # --- download mode ---
    if not args.download_dir:
        eprint('Error: download directory not provided. Run with --help for usage')
        return 2

    download_dir = ub.Path(args.download_dir)

    # Iterate benchmarks and versions
    final_ret = 0
    for benchmark in benchmark_list:
        # Determine versions per benchmark (may match multiple when selector is a MultiPattern)
        if version_arg in {'latest', 'auto'}:
            eprint(
                f"Resolving latest version for benchmark '{benchmark}' (backend={args.backend})..."
            )
        version_list = resolve_versions(benchmark, version_arg)
        if not version_list:
            if version_arg in {'latest', 'auto'}:
                eprint(
                    f"Error: could not determine latest version for benchmark '{benchmark}' (no runs found?)."
                )
                final_ret = 1
                if args.stop_on_error:
                    return final_ret
                continue
            else:
                eprint(
                    f"Warning: no versions matched selector '{version_arg}' for benchmark '{benchmark}'"
                )
                continue

        if version_arg in {'latest', 'auto'}:
            eprint(f'Using latest version for {benchmark}: {version_list[0]}')

        for version in version_list:
            bucket_base = storage._runs_root(benchmark)
            src = f'{bucket_base}/{version}'
            dest_root = download_dir / benchmark / 'benchmark_output' / 'runs'
            dest = dest_root / version

            print(f'HELM benchmark: {benchmark}')
            print(f'Version:        {version}')
            print(f'Source:         {src}')
            print(f'Destination:    {dest}')
            print()

            ret = _do_requested_download(
                storage, benchmark, version, dest, verbose, runs, checksum
            )
            if ret != 0:
                final_ret = ret
                if args.stop_on_error:
                    return final_ret

    return final_ret

__cli__ = DownloadHelmConfig
__cli__.main = main

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))