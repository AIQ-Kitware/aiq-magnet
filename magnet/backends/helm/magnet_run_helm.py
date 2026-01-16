from importlib.metadata import entry_points
from helm.benchmark.run import main as helm_run_main


def load_run_specs():
    # Each entrypoint under the 'helm.run_specs' group should be a
    # module that includes HELM run_specs (defined using the HELM
    # run_spec decorator)
    for ep in entry_points(group='helm.run_specs'):
        ep.load()

def main():
    load_run_specs()
    helm_run_main()

if __name__ == "__main__":
    main()
