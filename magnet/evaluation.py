import builtins
from graphlib import TopologicalSorter
from typing import Any, Dict, List, Tuple, get_origin, get_args
import yaml
import argparse


def _check_arbitrary_code(code_str: str) -> None:
    pass

class EvaluationCard:
    """
    Specification of an empirical claim with resolvable symbols and metadata

    Example:
        >>> from magnet.evaluation import EvaluationCard
        >>> card = EvaluationCard("simple.yaml")
        >>> card.evaluate()
        VERIFIED
    """
    def __init__(self, path):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.title = cfg.get("title", "")
        self.description = cfg.get("description", "")

        self.claim = Claim(cfg.get("claim"))
        self.symbols = Symbols(cfg.get("symbols", {}))

    def evaluate(self) -> str:
        """
        Run the evaluation specification

        1. Resolve symbol definitions
        2. Evaluate claim under symbol values
        """
        # Could log requests from here (i.e. timestamps), I think this was done in other code segments
        self.symbols.resolve()
        self.claim.evaluate(self.symbols())
        return self.claim.status

    def summarize(self):
        """
        Human-readable summary of card in its current state
        """
        print(f"Title:       {self.title}")
        print(f"Description: {self.description}")
        print("================================")
        print(f"SYMBOLS:     {self.symbols()}")
        print(f"CLAIM:       \n{self.claim}")
        print("================================")
        print(f"STATUS:      {self.claim.status}""")

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
            exec(self.claim, symbols)
            self.status = "VERIFIED"
        except AssertionError as e:
            self.status = "FALSIFIED"
            print(f"Assertion does not hold: {e}")
        except NameError as e:
            self.status = "INCONCLUSIVE"
            # This doesn't guarantee the missing variable is a symbol
            print(f"SymbolNotResolved: {e}")
        except Exception as e:
            self.status = "INCONCLUSIVE"
            print(f"ERROR evaluating claim: {e}")

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
    Dictionary of Symbols used as context for claim

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

    def resolve(self):
        """
        Trace dependency graph to resolve each symbol definition

        Values stored in Symbol instances
        """
        symbol_definitions = {}
        for symbol in self._construct_dependency_order():
            symbol_definitions[symbol] = self.symbols[symbol].eval(symbol_definitions.copy())

    def _construct_dependency_order(self) -> List[Symbol]:
        """
        Construct dependency order
        """
        dependency_graph = {symbol: instance.dependencies for symbol, instance in self.symbols.items()}
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


def main():
    parser = build_parser()
    args = parser.parse_args()

    card = EvaluationCard(args.path)
    card.evaluate()
    card.summarize()

if __name__ == "__main__":
    main()
