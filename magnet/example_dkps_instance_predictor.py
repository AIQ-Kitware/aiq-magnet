import random
import argparse
import pandas as pd

from magnet.instance_predictor import InstancePredictor, InstancePrediction
from magnet.data_splits import TrainSplit, SequesteredTestSplit

from sklearn.linear_model import LinearRegression, LogisticRegression
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

class DKPSInstancePredictor(InstancePredictor):
    def __init__(
        self,
        num_example_runs: int = 3,
        num_eval_samples: int = 20,
        random_seed: int = 1,
        n_components_cmds: int = 8,
        dataset: str = "med_qa",
        metric: str = "exact_match",
    ):
        super().__init__(
            num_example_runs = num_example_runs, 
            num_eval_samples = num_eval_samples, 
            random_seed      = random_seed
        )
        self.n_components_cmds = n_components_cmds
        self.dataset = dataset
        self.metric = metric

    def run_spec_filter(self, run_spec):
        return run_spec['name'].startswith(self.dataset)

    def predict(self,
        train_split: TrainSplit,
        sequestered_test_split: SequesteredTestSplit
    ) -> list[InstancePrediction]:
        
        # --
        # Parse MAGNET format
        
        # Unpack split classes into dataframes
        train_run_specs_df       = train_split.run_specs
        train_scenario_states_df = train_split.scenario_state
        # train_stats_df           = train_split.stats
        train_instance_stats_df   = train_split.per_instance_stats

        # eval_run_specs_df = sequestered_test_split.run_specs  # NOQA
        eval_scenario_state_df = sequestered_test_split.scenario_state

        # --
        # Convert to our format
        
        id2magnet = eval_scenario_state_df[['scenario_state.request_states.instance.id', 'magnet.instance_predict_id']]
        id2magnet = id2magnet.set_index('scenario_state.request_states.instance.id').to_dict()['magnet.instance_predict_id']
        
        metrics = train_run_specs_df['run_spec.metric_specs'].iloc[0][0]['args']['names']

        def _fmt_df(scenario_states_df, instance_stats_df=None):
            df = scenario_states_df[[
                'run_spec.name',
                'scenario_state.adapter_spec.model',
                'scenario_state.request_states.instance.id',
                'scenario_state.request_states.result.completions'
            ]].copy()

            df['model_family'] = df['scenario_state.adapter_spec.model'].apply(lambda x: x.split('/')[0])
            df['model']        = df['scenario_state.adapter_spec.model'].apply(lambda x: x.split('/')[-1])
            df['response']     = df['scenario_state.request_states.result.completions'].apply(lambda x: x[0]['text']) 

            df = df.rename(columns={
                'run_spec.name' : 'run_spec',
                'scenario_state.request_states.instance.id': 'instance_id',
            })

            if instance_stats_df is not None:
                df_stats = instance_stats_df[instance_stats_df['per_instance_stats.stats.name.name'].isin(metrics)]
                df_stats = df_stats.pivot(
                    index   = ['run_spec.name', 'per_instance_stats.instance_id'],
                    columns = 'per_instance_stats.stats.name.name',
                    values  = 'per_instance_stats.stats.mean'
                ).reset_index()
                
                df_stats = df_stats.rename(columns={
                    'run_spec.name' : 'run_spec',
                    'per_instance_stats.instance_id': 'instance_id'
                })

                df = pd.merge(df, df_stats, on=['run_spec', 'instance_id'], how='left')
            
            df = df.sort_values(['model', 'instance_id']).reset_index(drop=True)
            
            cols = ['run_spec', 'instance_id', 'model_family', 'model', 'response']
            if instance_stats_df is not None:
                cols += metrics
            
            return df[cols]

        df_train = _fmt_df(train_scenario_states_df, train_instance_stats_df)
        df_valid = _fmt_df(eval_scenario_state_df)
        
        # --
        # Data checks
        
        instance_ids = df_train.instance_id.unique()
        assert (instance_ids == df_valid.instance_id.unique()).all(), 'All instance_ids in the valid set must be in the train set'
        
        train_models    = df_train.model.unique()
        target_model    = df_valid.model.unique()[0]
        target_run_spec = df_valid[df_valid.model == target_model].run_spec.unique()[0]
        
        assert df_valid[df_valid.model == target_model].run_spec.unique().shape[0] == 1, 'Only one target run_spec is supported'
        assert df_valid.model.unique().shape[0] == 1, 'Only one target model is supported'
        assert target_model not in train_models, 'Target model must be different from train models'
        
        assert df_valid.instance_id.isin(df_train.instance_id.unique()).all(), \
            'All instance_ids in the valid set must be in the train set'

        # --
        # Compute embeddings
        
        df_train = _compute_embeddings(df_train, embed_model='onehot') # [TODO] add support for other embedders
        df_valid = _compute_embeddings(df_valid, embed_model='onehot')

        embedding_dict = _make_embedding_dict(pd.concat([df_train, df_valid]))
        P              = DKPS(n_components_cmds=self.n_components_cmds).fit_transform(embedding_dict, return_dict=True)

        # [TODO] filter models from same model family as target model

        X_train = np.vstack([P[m] for m in train_models])
        X_valid = P[target_model][None]

        predictions = []
        for instance_id in instance_ids:
            for metric in metrics:
                df_train_sub = df_train[df_train.instance_id == instance_id]
                assert (df_train_sub.model.values == train_models).all(), 'df_train_sub.model.values != train_models'
                
                y_train = df_train_sub[metric].values
                
                # [TODO] what if y_train only has one value?
                #        could either assume - "everything is right" or "does it match target model response"
                
                
                is_binary = (np.unique(y_train) == [0, 1]).all()
                if is_binary:
                    # if the target is binary (via a hacky check), use LogisticRegression
                    lr      = LogisticRegression().fit(X_train, y_train)
                    y_hat   = lr.predict_proba(X_valid)[0][1]
                else:
                    # otherwise, use LinearRegression
                    lr      = LinearRegression().fit(X_train, y_train)
                    y_hat   = lr.predict(X_valid)[0]
                
                predictions.append(
                    InstancePrediction(
                        run_spec_name       = target_run_spec,
                        instance_predict_id = id2magnet[instance_id],
                        stat_name           = metric,
                        mean                = y_hat
                    )
                )

        return predictions

if __name__ == "__main__":
    import numpy as np
    np.random.seed(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-example-runs", default=50, type=int, help="Number of training runs used by DKPS.",
    )
    parser.add_argument(
        "--num-eval-samples", default=10, type=int, help="Number of queries used by DKPS.",
    )
    parser.add_argument("--seed", default=1, type=int, help="Random seed to use.")
    
    parser.add_argument("--n-components-cmds", default=8, type=int, help="Number of components used by DKPS.")
    
    args = parser.parse_args()

    predictor = DKPSInstancePredictor(
        random_seed       = args.seed,
        num_example_runs  = args.num_example_runs,
        num_eval_samples  = args.num_eval_samples,
        n_components_cmds = args.n_components_cmds,
    )

    predictor(
        '/Users/bjohnson/data/crfm-helm-public/lite/benchmark_output',
        "_all" # [FEATURE-REQUEST] I wanted to use all runs, from all suites. I don't think there's a flag for that, so I had to copy all `v*` to `_all` on filesystem
    )
