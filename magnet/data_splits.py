class DataSplit:
    """
    Enapsulates data for a particualr data split

    Attributes:
        run_specs: DataFrame | None
        scenario_state: DataFrame | None
        stats: DataFrame | None
    """
    def __init__(self, run_specs=None, scenario_state=None, stats=None, per_instance_stats=None):
        self.run_specs = run_specs
        self.scenario_state = scenario_state
        self.stats = stats
        self.per_instance_stats = per_instance_stats


class TrainSplit(DataSplit):
    ...


class TestSplit(DataSplit):

    def sequester(self):
        """
        Drop the results for components that should not have access to it.
        """
        sequestered_split = SequesteredTestSplit(
            run_specs=self.run_specs,
            scenario_state=self.scenario_state
        )
        return sequestered_split


class SequesteredTestSplit(TestSplit):
    def __init__(self, run_specs=None, scenario_state=None, stats=None, per_instance_stats=None):
        assert stats is None, 'cannot specify stats here'
        assert per_instance_stats is None, 'cannot specify per_instance_stats here'
        super().__init__(run_specs=run_specs, scenario_state=scenario_state)
