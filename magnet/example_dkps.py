"""
    example_dkps.py
    
    Run:
        python magnet/example_dkps.py
"""

import argparse
import numpy as np
import pandas as pd
from helm.benchmark.metrics.statistic import Stat

from magnet.predictor import Predictor, TrainSplit, SequesteredTestSplit

from sklearn.linear_model import LinearRegression
from dkps.dkps import DataKernelPerspectiveSpace as DKPS

# --
# Helpers

def _onehot_embedding(df, dataset):
    if dataset == 'med_qa':
        lookup = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3}
        
        embeddings = np.zeros((len(df), 4))
        for i, xx in enumerate(df.response.values):
            xx = xx.strip().upper()
            if xx in lookup:
                embeddings[i, lookup[xx]] = 1
        
        df['embedding'] = embeddings.tolist()
    
    elif 'legalbench' in dataset:
        raise NotImplementedError("!! [TODO] legalbench preprocessing not implemented yet !!")
        
        # slightly different - bad values get mapped to 0
        n_levels   = len(df.response.unique())
        embeddings = np.zeros((len(df), n_levels))
        for i, xx in enumerate(df.response.values):
            embeddings[i, xx] = 1

        df['embedding'] = embeddings.tolist()
    else:
        raise ValueError(f'{dataset} is not supported for onehot embeddings')
    
    return df

def _compute_embeddings(df, embed_model=None):
    if embed_model == 'onehot':
        df = _onehot_embedding(df, dataset="med_qa") # [TODO] add support for other datasets
    else:
        raise NotImplementedError("!! [TODO] non-onehot embeddings not implemented yet !!")
    
    return df

def _make_embedding_dict(df):
    model_names  = df.model.unique()
    instance_ids = df.instance_id.unique()
    
    embedding_dict = {}
    for model_name in model_names:
        sub = df[df.model == model_name]
        assert (sub.instance_id.values == instance_ids).all(), f'instance_ids are not the same for model {model_name}'
        embedding_dict[model_name] = np.vstack(sub.embedding.values)
    
    embedding_dict = {k:v[:,None] for k,v in embedding_dict.items()}
    
    return embedding_dict


# --
# Predictor

class DKPSPredictor(Predictor):
    def __init__(
        self,
        num_example_runs: int = 3,
        num_eval_samples: int = 20,
        random_seed: int = 1,
        n_components_cmds: int = 8,
    ):
        super().__init__(num_example_runs, num_eval_samples, random_seed)
        self.n_components_cmds = n_components_cmds
    
    def run_spec_filter(self, run_spec):
        return run_spec['name'].startswith("med_qa") # [TODO] this code _should_ work for all datasets, modulo dataset-specific preprocessing

    def predict(
        self,
        train_split: TrainSplit,
        sequestered_test_split: SequesteredTestSplit
    ) -> dict[str, list[Stat]]:
        
        # Unpack split classes into dataframes
        train_run_specs_df       = train_split.run_specs
        train_scenario_states_df = train_split.scenario_state
        train_stats_df           = train_split.stats

        # eval_run_specs_df        = sequestered_test_split.run_specs
        eval_scenario_state_df  = sequestered_test_split.scenario_state
                
        # <<
        # [HACK] Filter data - pick one of the splits (I don't know what splits are supposed to mean in HELM?)
        splits = eval_scenario_state_df['scenario_state.request_states.instance.split'].unique()
        split  = "test"
        assert split in splits
        assert split in train_scenario_states_df['scenario_state.request_states.instance.split'].unique()
        # <<
        
        predicted_stats = {}
        metrics = train_run_specs_df['run_spec.metric_specs'].iloc[0][0]['args']['names']
        for metric in metrics:

            _train_scenario_states_df = train_scenario_states_df[train_scenario_states_df['scenario_state.request_states.instance.split'] == split]
            _eval_scenario_state_df   = eval_scenario_state_df[eval_scenario_state_df['scenario_state.request_states.instance.split'] == split]
            
            _train_stats_df = train_stats_df[(
                (train_stats_df['stats.name.name'] == metric) &
                (train_stats_df['stats.name.perturbation.computed_on'].isnull()) &
                (train_stats_df['stats.name.split'] == split)
            )]
            
            assert _train_stats_df['run_spec.name'].nunique() == _train_stats_df.shape[0]
            
            # --
            # Format data

            def _fmt_df(df_raw):
                df = df_raw[[
                    'run_spec.name',
                    'scenario_state.adapter_spec.model',
                    'scenario_state.request_states.instance.id',
                    'scenario_state.request_states.result.completions'
                ]].copy()
                
                df.columns         = ['run_spec', 'model', 'instance_id', 'response']
                df['model_family'] = df.model.apply(lambda x: x.split('/')[0])
                df.model           = df.model.apply(lambda x: x.split('/')[-1])
                df.response        = df.response.apply(lambda x: x[0]['text']) 
                
                df = df[['run_spec', 'model_family', 'model', 'instance_id', 'response']]
                
                df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)
                
                return df

            df_train = _fmt_df(_train_scenario_states_df)
            df_valid = _fmt_df(_eval_scenario_state_df)
            
            # drop training instances that are not in the eval set
            #   - this is not a hack - it's how the algorithm works - but we should record that we aren't using all of the training data
            df_train = df_train[df_train.instance_id.isin(df_valid.instance_id)]
            assert (df_train.instance_id.unique() == df_valid.instance_id.unique()).all()

            # [TODO] make sure that none of the training models are in the same family as the eval model
            # - IMO this should be done outside of predictor
            
            # --
            # Preprocess responses
            #
            # [TODO] preprocess response - this is dataset dependent.  See:
            # - https://github.com/jataware/dkps/blob/main/examples/helm/parsers/legalbench.py
            # - https://github.com/jataware/dkps/blob/main/examples/helm/parsers/med_qa.py
            
            # --
            # Compute embeddings
            # [TODO] add support for other embedders
            
            df_train = _compute_embeddings(df_train, embed_model='onehot')
            df_valid = _compute_embeddings(df_valid, embed_model='onehot')

            # --
            # Get scores for train models
            
            df_score         = _train_stats_df[['run_spec.name', 'stats.mean']]
            df_score.columns = ['run_spec', 'score']
            df_score         = pd.merge(df_score, df_train[['run_spec', 'model']].drop_duplicates(), on='run_spec', how='left')
            assert set(df_score['run_spec'].unique()) == set(df_train['run_spec'].unique())
            model2score = df_score.set_index('model')['score'].to_dict()
            
            embedding_dict = _make_embedding_dict(pd.concat([df_train, df_valid]))
            P              = DKPS(n_components_cmds=self.n_components_cmds).fit_transform(embedding_dict, return_dict=True)
            
            X_train = np.vstack([P[m] for m in df_train.model.unique()])
            y_train = np.array([model2score[m] for m in df_train.model.unique()])
            
            for row in df_valid[['run_spec', 'model']].drop_duplicates().itertuples():
                
                X_test  = P[row.model][None]
                lr      = LinearRegression().fit(X_train, y_train)
                y_hat   = lr.predict(X_test)[0]
                
                predicted_stats.setdefault(row.run_spec, []).append(
                    Stat(
                        **{
                            'name'        : {'name' : metric, 'split' : split},
                            'count'       : 1,
                            'sum'         : y_hat,
                            'sum_squared' : y_hat**2,
                            'min'         : y_hat,
                            'max'         : y_hat,
                            'mean'        : y_hat,
                            'variance'    : 0.0,
                            'stddev'      : 0.0,
                        }
                    )
                )

        return predicted_stats


if __name__ == "__main__":
    import numpy as np
    np.random.seed(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-example-runs", default=50, type=int, help="Number of training runs used by DKPS.",
    )
    parser.add_argument(
        "--num-eval-samples", default=4, type=int, help="Number of queries used by DKPS.",
    )
    parser.add_argument("--seed", default=1, type=int, help="Random seed to use.")
    
    parser.add_argument("--n-components-cmds", default=8, type=int, help="Number of components used by DKPS.")
    
    args = parser.parse_args()

    predictor = DKPSPredictor(
        random_seed       = args.seed,
        num_example_runs  = args.num_example_runs,
        num_eval_samples  = args.num_eval_samples,
        n_components_cmds = args.n_components_cmds,
    )

    predictor(
        '/Users/bjohnson/data/crfm-helm-public/lite/benchmark_output',
        "_all" # [FEATURE-REQUEST] I wanted to use all runs, from all suites. I don't think there's a flag for that, so I had to copy all `v*` to `_all` on filesystem
    )