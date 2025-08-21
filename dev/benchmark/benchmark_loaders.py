from magnet.helm_outputs import HelmRun
import ubelt as ub
import timerit
from typing import Iterator


def main():
    run = HelmRun.demo()

    variants = {
        'msgspec': run.msgspec,
        'json': run.json,
        '_json_stdlib': run._json_stdlib,
        '_json_ujson': run._json_ujson,
        '_json_orjson': run._json_orjson,
        'dataclass': run.dataclass,
        'dataframe': run.dataframe,
    }

    methods = [
        'stats',
        'run_spec',
        'scenario_state',
        'per_instance_stats',
    ]

    for name in methods:
        ti = timerit.Timerit(100, bestof=10, verbose=2)

        outputs = {}
        for key, value in variants.items():
            func = getattr(value, name)
            for timer in ti.reset(f'Load {name=} with {key=}'):
                output = func()
                if isinstance(output, Iterator):
                    output = list(output)
                outputs[key] = output
        print(f'ti.measures = {ub.urepr(ti.measures, nl=2, precision=8, align=':')}')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/magnet-sys-exploratory/dev/benchmark/benchmark_loaders.py
    """
    main()
