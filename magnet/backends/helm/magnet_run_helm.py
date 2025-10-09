from helm.benchmark.run import main as helm_run_main

# Import our custom run_specs so that they're hooked up for the HELM
# run
from magnet.backends.helm import run_specs as _unused


def main():
    helm_run_main()

if __name__ == "__main__":
    main()
