import builtins
from graphlib import TopologicalSorter
from itertools import product
from typing import Any, Dict, List, Self, Tuple, get_origin, get_args
import yaml
import argparse
import sys

import scriptconfig as scfg
from rich import print


class EvaluationConfig(scfg.DataConfig):
    """
    Resolve an Evaluation Card
    """
    __epilog__ = """
    Usage:
      ./evaluation.py <evaluation_card_path>

    Examples:
      # Show docs
      python -m magnet.evaluation --help

      # Run example card
      python -m magnet.evaluation magnet/cards/simple.yaml
    """

    path = scfg.Value(
        None, required=True, position=1, help='Path to evaluation card YAML'
    )


class EvaluationCard:
    """
    Specification of an empirical claim with resolvable symbols and metadata

    Example:
        >>> from importlib.resources import files
        >>> from magnet.evaluation import EvaluationCard
        >>> card_name = 'simple.yaml'
        >>> card_path = files('magnet') / 'cards' / card_name
        >>> card = EvaluationCard(card_path)
        >>> card.evaluate()
        VERIFIED
    """
    def __init__(self, path):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.title = cfg.get("title", "")
        self.description = cfg.get("description", "")

        self.claim = Claim(cfg.get("claim"))
        self.symbols = cfg.get("symbols", {})

        self.evaluations = []

    def status(self) -> str:
        """
        Declaration of card state, whether not started, in progress, or complete
        """
        if len(self.evaluations) == 0:
            return "UNEVALUTED"

        not_evaluated_count = sum([evaluation.claim.status == "UNVERIFIED" for evaluation in self.evaluations])
        percent_not_evaluated = not_evaluated_count / len(self.evaluations)

        if percent_not_evaluated == 0:
            return "EVALUATED"
        else:
            return f"{percent_not_evaluated:.2f} REMAINING"

    def evaluate(self):
        """
        Run the evaluation specification

        1. Resolve symbol definitions
        2. Evaluate claim under symbol values
        3. Summarize general finding
        """
        self.evaluations = self.dispatch(Symbols.decompose_symbol_defs(self.symbols))

        results = []
        for evaluation in self.evaluations:
            status, _ = evaluation.execute()
            results.append(status)

        total = len(self.evaluations)
        percentage = lambda count: count / total

        verified_count = results.count('VERIFIED')
        falsified_count = results.count('FALSIFIED')
        inconclusive_count = results.count('INCONCLUSIVE')

        print("================================")
        print(f"Settings Evaluated: {total}")
        print(f"  Verified:     {percentage(verified_count):.2f}")
        print(f"  Falsified:    {percentage(falsified_count):.2f}")
        print(f"  Inconclusive: {percentage(inconclusive_count):.2f}")
        print("================================")
        print('\n')

        card_result = ''
        if falsified_count:
            card_result = 'FALSIFIED'
        elif inconclusive_count:
            card_result = 'INCONCLUSIVE'
        else:
            card_result = 'VERIFIED'

        self.claim.status = card_result
        return card_result

    def dispatch(self, flattened_sweep): #: List[Symbols]) -> List[EvaluationTask]:
        return [EvaluationTask(Claim({'python': self.claim.claim}), symbols) for symbols in flattened_sweep]

    def summarize(self):
        """
        Human-readable summary of card in its current state
        """
        print(f"[bold]Title:[/bold]       {self.title}")
        print(f"[bold]Description:[/bold] {self.description}")
        print("================================")
        #print(f"SYMBOLS:     {self.symbols()}")
        print(f"[bold]CLAIM:[/bold]       \n{self.claim}")

        status = self.status()
        if self.claim.status == 'VERIFIED':
            claim_status_color = "green"
        elif self.claim.status == 'FALSIFIED':
            claim_status_color = "red"
        else:
            claim_status_color = "yellow"

        if status == 'EVALUATED':
            print("================================")
            print(f"[bold]RESULT:[/bold]      [bold][{claim_status_color}]{self.claim.status}[/{claim_status_color}][/bold]""")

        print("================================")
        print(f"[bold]CARD STATUS:[/bold] {status}""")

class EvaluationTask:
    """
    Singular submission from an Evaluation Card
    """
    def __init__(self, claim, symbols):
        self.claim = claim
        self.symbols = symbols

    def execute(self) -> Tuple[str, str]:
        self.symbols.resolve()
        # x -> y -> z1 -> a1 -> res1
        #           ...
        #           zn -> an -> resn
        # make sure x,y are done once / before sweep
        return self.claim.evaluate(self.symbols())

    def record_run(self):
        # Could log requests from here (i.e. timestamps), I think this was done in other code segments
        # timestamp, symbol value, claim result
        raise NotImplementedError

class Claim:
    """
    Represents a verifiable assertion for a set of resolved symbols

    ***
    Currently assumes
    1. claim is valid and safe python code
    2. all symbols can be resolved from card
    3. No additional dependencies are needed
    4. Any conclusions drawn are as reliable as claim itself (i.e. verification is strictly: 'does code execute without error')
    ***

    Example:
        >>> from magnet.evaluation import Claim
        >>> self = Claim({'python': "assert x + 2 == 4"})
        >>> print(self)
        assert x + 2 == 4
        >>> self.evaluate({'x': 2})
        >>> print(self.status)
        VERIFIED
    """
    def __init__(self, raw):
        self.claim = raw.get('python')
        self.status = "UNVERIFIED"

    def evaluate(self, symbols: Dict[str, Any]={}):
        """
        Execute the claim subject to symbols definitions

        if True:
            VERIFIED
        elif AssertionError:
            FALSIFIED
        else:
            INCONCLUSIVE
        """
        try:
            out_msg = ""
            exec(self.claim, symbols)
            self.status = "VERIFIED"
        except AssertionError as e:
            self.status = "FALSIFIED"
            out_msg = f"Assertion does not hold: {e}"
        except NameError as e:
            self.status = "INCONCLUSIVE"
            # This doesn't guarantee the missing variable is a symbol
            out_msg = f"SymbolNotResolved: {e}"
        except Exception as e:
            self.status = "INCONCLUSIVE"
            out_msg = f"ERROR evaluating claim: {e}"
        finally:
            if out_msg:
                print(out_msg)
            return self.status, out_msg

    def __repr__(self) -> str:
        return self.claim

class Symbol:
    """
    Single resolvable unit of a claim

    Example:
        >>> from magnet.evaluation import Symbol
        >>> x = Symbol('x', {'type': "List[int]", 'python': "x = [10]"})
        >>> x.eval()
        [10]
    """
    def __init__(self, name, spec):
        self.name = name
        self.value = spec.get('value')
        self.sweep = spec.get('sweep')
        self.type = spec.get('type', 'List[int]')
        self.definition = spec.get('python', '')
        self.dependencies = spec.get('depends_on', [])

    def eval(self, context: Dict[str, Any]= {}) -> Any:
        """
        Resolve symbol definition

        FIXME: type verification is currently limited and hacky
        """
        if self.value is None:
            print(f"Resolving: {self.name}")
            exec(self.definition, context)
            if self._check_type(self.type, context[self.name]):
                self.value = context[self.name]
            else:
                raise TypeError(f'{self.name}: {context[self.name]} is not {self.type}')

        return self.value

    def _check_type(self, type_str, value) -> bool:
        """
        Validate value is of type str_type
        """
        # TODO: static 'vocabulary' of allowable types / support more than List[Any], Dict[str, Any]
        str_to_type = {'List': List, 'Dict': Dict, 'Tuple': Tuple, 'Any': Any}
        type = eval(type_str, str_to_type)
        return self._check_collections(type, value)

    def _check_collections(self, target_type, value):
        """
        Recursively evaluate if value is target_type
        """
        collection_type = get_origin(target_type)
        members = get_args(target_type)

        match collection_type:
            case builtins.list:
                if isinstance(value, list):
                    return all(self._check_collections(members[0], entry) for entry in value)
            case builtins.dict:
                if isinstance(value, dict):
                    return all(
                        self._check_collections(members[0], key_entry) and self._check_collections(members[1], value_entry)
                        for key_entry, value_entry in value.items()
                    )
            case builtins.tuple:
                if isinstance(value, tuple) and len(value) == len(members):
                    return all(self._check_collections(type, val) for type, val in zip(members, value))
            case None:
                # Any or primative
                return target_type is Any or isinstance(value, target_type)
            case _:
                return False

class Symbols:
    """
    Collection of Symbol configurations used as context for claim

    Example:
        >>> from magnet.evaluation import Symbols
        >>> symbols = Symbols({'x': {'type': "List[int]", 'python': "x = [10]"}})
        >>> symbols()
        {'x': None}
        >>> symbols.resolve()
        >>> symbols()
        {'x': [10]}
    """
    def __init__(self, symbol_specs) -> None:
        self.symbols = {symbol: Symbol(symbol, definition) for symbol, definition in symbol_specs.items()}

    @classmethod
    def decompose_symbol_defs(cls, symbol_definitions) -> List[Self]:
        """
        Flatten sweep values into a list of resolvable Symbols
        """
        configurations = []
        aggregate_configuration = cls(symbol_definitions)

        sweep_symbols = aggregate_configuration._find_sweep_symbols()
        if sweep_symbols:
            sweep_values = [sweep.sweep for sweep in sweep_symbols]
            combinations = product(*sweep_values)

            for combo in combinations:
                sweep_fill = dict(zip([symbol.name for symbol in sweep_symbols], combo))
                flattened_symbols = cls(symbol_definitions)
                for k,v in sweep_fill.items():
                    flattened_symbols.symbols[k].value = v
                configurations.append(flattened_symbols)
        else:
            configurations.append(aggregate_configuration)

        return configurations

    def resolve(self):
        """
        Trace dependency graph to resolve each symbol definition

        Values stored in Symbol instances
        """
        symbol_definitions = {}

        for symbol in self._construct_dependency_order():
            symbol_definitions[symbol] = self.symbols[symbol].eval(symbol_definitions.copy())

    def _find_sweep_symbols(self) -> List[Symbol]:
        return [symbol for symbol in self.symbols.values() if symbol.sweep]

    def _construct_dependency_order(self) -> List[Symbol]:
        """
        Construct dependency order
        """
        dependency_graph = {name: symbol.dependencies for name, symbol in self.symbols.items()}
        sorter = TopologicalSorter(dependency_graph)
        return list(sorter.static_order())

    def __call__(self):
        return {symbol: self.symbols[symbol].value for symbol in self.symbols}


def build_parser():
    parser = argparse.ArgumentParser(description="Resolve an Evaluation Card")

    parser.add_argument('path',
                        type=str,
                        help="Path to evaluation card YAML file")

    return parser


def main(argv=None, **kwargs):
    args = EvaluationConfig.cli(
        argv=argv, data=kwargs, strict=True, verbose='auto', special_options=False
    )

    card = EvaluationCard(args.path)
    card.evaluate()
    card.summarize()


__cli__ = EvaluationConfig
__cli__.main = main

if __name__ == '__main__':
    main(sys.argv[1:])
