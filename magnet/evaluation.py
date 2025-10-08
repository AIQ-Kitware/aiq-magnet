from graphlib import TopologicalSorter
from typing import Any, Dict, List, get_origin, get_args
import yaml 

class EvaluationCard:
    """
    Specification of an empirical claim with resolvable symbols and metadata

    Example:
        >>> from magnet.evaluation import EvaluationCard
        >>> card = EvaluationCard("simple.yaml")
        >>> card.evaluate()
        VERIFIED
    """
    def __init__(self, name):
        # load evaluation card from 'magnet/cards' (may want to FIXME to path)
        with open(f'magnet/cards/{name}', 'r') as f:
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
        # Could log requests from here
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
        self.value = None
        self.type = spec.get('type', 'List[int]')
        self.definition = spec.get('python', '')
        self.dependencies = spec.get('dependencies', [])
    
    def eval(self, context: Dict[str, Any]= {}) -> Any:
        """
        Resolve symbol definition

        FIXME: type verification is currently limited and hacky
        """
        exec(self.definition, context)
        if self._check_type(self.type, context[self.name]):
            self.value = context[self.name]
        return self.value
    
    def _check_type(self, str_type, value) -> bool:
        """
        Validate value is of type str_type

        # TODO: support more than List[Any]
        """
        str_to_type = {'List': List}
        type = eval(str_type, str_to_type)
        print(get_args(type))
        if get_origin(type) is list:
            if isinstance(value, list):
                return all(isinstance(entry, get_args(type)[0]) for entry in value)
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