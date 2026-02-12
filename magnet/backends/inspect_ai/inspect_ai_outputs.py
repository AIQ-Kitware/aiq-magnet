from __future__ import annotations

import ubelt as ub
from inspect_ai.analysis import (
    evals_df, samples_df, messages_df, events_df
)
from inspect_ai.log import list_eval_logs
from pandas import DataFrame


class InspectAIOutputs(ub.NiceRepr):
    """
    Class to represent and explore Inspect AI outputs
    """
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str | PathLike):
                The benchmark output directory containing a runs folder with
                multiple suites.
        """
        self.root_dir = root_dir

    def eval_logs(self, pattern=None, filter=None) -> InspectAIEvalLogs:
        if filter is not None:
            return InspectAIEvalLogs(list_eval_logs(self.root_dir, filter=filter))
        elif pattern is not None:
            return InspectAIEvalLogs(self.root_dir.glob(pattern))
        else:
            return InspectAIEvalLogs(list_eval_logs(self.root_dir))

    def __nice__(self):
        return self.root_dir


class InspectAIEvalLogs(ub.NiceRepr):
    """
    Represents a single log in a set of Inspect AI outputs.
    """
    def __init__(self, logs):
        self.logs = logs

    def evals(self) -> DataFrame:
        """
        Evaluation level data (e.g. task, model, scores, etc.). One
        row per log.
        """
        return evals_df(self.logs)

    def samples(self) -> DataFrame:
        """
        Sample level data (e.g. input, metadata, scores, errors, etc.)
        One row per sample, where each log contains many samples.
        """
        return samples_df(self.logs)

    def messages(self) -> DataFrame:
        """
        Message level data (e.g. role, content, etc.). One row per
        message, where each sample contains many messages.
        """
        return messages_df(self.logs)

    def events(self) -> DataFrame:
        """
        Event level data (type, timing, content, etc.). One row per
        event, where each sample contains many events.
        """
        return events_df(self.logs)
