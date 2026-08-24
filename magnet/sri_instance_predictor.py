# import random
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from magnet.data_splits import SequesteredTestSplit, TrainSplit
from magnet.instance_predictor import InstancePrediction, InstancePredictor

# ---------------------------
# CLIP embedding helper
# ---------------------------

_CLIP_MODEL = None
_CLIP_TOKENIZER = None
_CLIP_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def clip_embedding(
    texts, batch_size=32, model_id='openai/clip-vit-base-patch32'
):
    """
    Compute L2-normalized CLIP text embeddings for a list of strings.
    Uses a cached global model/tokenizer.
    Returns a tensor [len(texts), D].
    """
    global _CLIP_MODEL, _CLIP_TOKENIZER

    if _CLIP_MODEL is None or _CLIP_TOKENIZER is None:
        _CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(model_id)
        _CLIP_MODEL = (
            CLIPModel.from_pretrained(model_id).eval().to(_CLIP_DEVICE)
        )

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = _CLIP_TOKENIZER(
                batch, return_tensors='pt', padding=True, truncation=True
            ).to(_CLIP_DEVICE)
            feats = _CLIP_MODEL.get_text_features(**inputs)  # [B, D]
            feats = F.normalize(feats, dim=-1)
            all_embs.append(feats.cpu())

    return torch.cat(all_embs, dim=0)  # [N, D]


# ---------------------------
# Ridge + conformal helpers
# ---------------------------


def ridge_closed_form(
    X: torch.Tensor,
    y: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ridge regression with unpenalized intercept via centering:
      w = argmin_w ||(X - meanX) w - (y - meany)||^2 + lam ||w||^2
      b = meany - meanX @ w
    X: (N, D), y: (N,)
    Returns: w: (D,), b: scalar
    """
    dtype0 = X.dtype
    X = X.double()
    y = y.double().view(-1, 1)
    N, D = X.shape

    mean_x = X.mean(dim=0, keepdim=True)
    mean_y = y.mean(dim=0, keepdim=True)
    Xc = X - mean_x
    yc = y - mean_y

    lam = float(max(lam, 1e-12))

    if D <= N:
        A = Xc.T @ Xc + lam * torch.eye(D, device=X.device, dtype=X.dtype)
        B = Xc.T @ yc
        w = torch.linalg.solve(A, B)  # (D,1)
    else:
        K = Xc @ Xc.T + lam * torch.eye(N, device=X.device, dtype=X.dtype)
        alpha = torch.linalg.solve(K, yc)  # (N,1)
        w = Xc.T @ alpha  # (D,1)

    b = mean_y - mean_x @ w  # (1,1)
    return w.view(-1).to(dtype0), b.view(-1).to(dtype0)[0]


@torch.no_grad()
def calibrate_factor_gauss(
    y: torch.Tensor, mu: torch.Tensor, lnvar: torch.Tensor, alpha: float
) -> float:
    """
    Split-conformal calibration factor for a Gaussian predictor:
      r_i = |y_i - mu_i| / std_i
      c = quantile_{1-alpha}(r)
    """
    var = lnvar.exp().clamp_min(1e-12)
    std = var.sqrt()
    r = (y - mu).abs() / (std + 1e-12)
    c = torch.quantile(r, 1 - alpha).item()
    return float(max(0.0, c))


class SRIInstancePredictor(InstancePredictor):
    """
    Example per-instance stat predictor using CLIP-based ridge + split conformal.

    Uses:
      - metrics_model_prompt_table.csv (columns: metric, model, prompt, performance, ...)
      - CLIP text encoder "openai/clip-vit-base-patch32" directly as the feature map.
    """

    def predict(
        self,
        train_split: TrainSplit,
        sequestered_test_split: SequesteredTestSplit,
    ) -> list[InstancePrediction]:
        # Unpack split classes into dataframes (not heavily used here)
        train_run_specs_df = train_split.run_specs  # NOQA
        train_scenario_states_df = train_split.scenario_state  # NOQA
        train_stats_df = train_split.stats  # NOQA
        # import xdev
        # xdev.embed()

        # Load the flat table you built previously
        # train_data_table = pd.read_csv('/home/joncrall/code/aiq-magnet/metrics_model_prompt_table.csv')

        # train_instance_stats = train_split.per_instance_stats
        # train_split.scenario_state['scenario_state.request_states.instance.input.text']
        # train_split.scenario_state['scenario_state.request_states.instance.id']
        # train_split.scenario_state['scenario_state.request_states.instance.input.text'].values

        # instances = train_split.instances

        # Enhance each instance with info from the run spec table.
        big_table = train_split.per_instance_stats.merge(
            train_split.run_specs, on='run_spec.name', how='left'
        )

        big_table2 = big_table.merge(
            train_scenario_states_df,
            on='magnet.instance_predict_id',
            how='left',
        )

        def make_object_spec_description(class_name: str, args) -> str:
            """
            Inverse of `parse_object_spec`:
            Given a class_name and args dict, return the description string.

            Format:
                <class_name>:<key>=<value>,<key>=<value>

            If args is empty, just returns <class_name>.
            """
            if not args:
                return class_name

            # Note: assumes values do not need escaping and will be parsed back as str/int/float
            arg_parts = [f'{key}={value}' for key, value in args.items()]
            return f'{class_name}:' + ','.join(arg_parts)

        scenario_components = big_table.prefix_subframe(
            'run_spec.scenario_spec', drop_prefix=True
        )
        datasets = []
        for _, row in scenario_components.iterrows():
            args = {k: v for k, v in row.to_dict().items() if not pd.isna(v)}
            class_name = args.pop('class_name')
            # Not sure if there is a helm-y way to construct the spec name,
            # this should work for now.
            description = make_object_spec_description(class_name, args)
            datasets.append(description)

        train_data_table_v2 = {}
        # run_spec.adapter_spec.model
        train_data_table_v2['metric'] = [
            x + '|test' for x in big_table['per_instance_stats.stats.name.name']
        ]  # do we need |test?
        train_data_table_v2['model'] = big_table[
            'run_spec.adapter_spec.model'
        ].values
        train_data_table_v2['prompt'] = big_table2[
            'scenario_state.request_states.instance.input.text'
        ].values
        train_data_table_v2['performance'] = big_table2[
            'per_instance_stats.stats.mean'
        ]
        train_data_table_v2['dataset'] = datasets
        train_data_table_v2['N'] = big_table2['per_instance_stats.stats.count']
        train_data_table_v2 = pd.DataFrame(train_data_table_v2)
        train_data_table = train_data_table_v2

        # Hyperparameters (can be tuned)
        lam_mean = 1e1
        lam_var = 1e1
        eps_var = 1e-2
        alpha = 0.05  # target 1 - alpha coverage

        def craft_predictor(
            df: pd.DataFrame,
            model_name: str,
            metric_name: str,
            banned_prompts: list[str],
        ):
            """
            - Filter df to rows for given (model, metric), excluding banned_prompts.
            - Randomly split filtered rows into 70% train, 30% cal.
            - Fit ridge on train for mean + log variance.
            - Use cal to compute split-conformal factor.
            - Return a function f(emb) -> (mean, lb, ub).
            """
            sub = df[
                (df['model'] == model_name) & (df['metric'] == metric_name)
            ].copy()
            if len(banned_prompts) > 0:
                sub = sub[~sub['prompt'].isin(banned_prompts)]

            sub = sub.dropna(subset=['prompt', 'performance'])
            if len(sub) < 5:
                # Not enough data; return a trivial constant predictor
                return lambda emb: (0.0, 0.0, 0.0)

            # Shuffle
            sub = sub.sample(frac=1.0, random_state=0).reset_index(drop=True)
            n = len(sub)
            n_train = max(1, int(0.7 * n))
            n_cal = max(1, int(0.3 * n))
            n_used = min(n, n_train + n_cal)
            n_train = min(n_train, n_used)
            n_cal = min(n_cal, n_used - n_train)
            print(f'ntrain {n_train} ncal {n_cal}')

            train_df = sub.iloc[:n_train]
            cal_df = sub.iloc[n_train : n_train + n_cal]

            # Collect unique prompts to embed once
            train_prompts = train_df['prompt'].tolist()
            cal_prompts = cal_df['prompt'].tolist()
            all_prompts = sorted(set(train_prompts + cal_prompts))
            prompt_to_idx = {p: i for i, p in enumerate(all_prompts)}

            embs = clip_embedding(all_prompts, batch_size=32)
            # Build X, y for train
            Xtr = torch.stack(
                [embs[prompt_to_idx[p]] for p in train_prompts], dim=0
            )
            ytr = torch.tensor(
                train_df['performance'].values, dtype=torch.double
            )

            # Build X, y for cal
            Xcal = torch.stack(
                [embs[prompt_to_idx[p]] for p in cal_prompts], dim=0
            )
            ycal = torch.tensor(
                cal_df['performance'].values, dtype=torch.double
            )

            # Fit mean ridge
            w1, b1 = ridge_closed_form(Xtr, ytr, lam=lam_mean)
            mu_tr = Xtr @ w1 + b1

            # Fit log-variance ridge
            r_tr = ytr - mu_tr
            t_tr = torch.log(r_tr.pow(2) + eps_var)
            w2, b2 = ridge_closed_form(Xtr, t_tr, lam=lam_var)

            # Calibrate on cal
            with torch.no_grad():
                mu_cal = Xcal @ w1 + b1
                lnvar_cal = Xcal @ w2 + b2
                cal_factor = calibrate_factor_gauss(
                    ycal, mu_cal, lnvar_cal, alpha=alpha
                )
                cal_factor = float(cal_factor)

            # Return predictor function
            def predictor(emb: torch.Tensor):
                """
                emb: 1D tensor [D] (CLIP embedding of a single prompt, already normalized).
                Returns: (mean, lb, ub)
                """
                with torch.no_grad():
                    # Match dtype and device of fitted weights
                    if emb.dim() == 1:
                        x = emb.view(1, -1)
                    else:
                        x = emb

                    x = x.to(w1.device).type_as(
                        w1
                    )  # ensure same dtype & device as w1/w2

                    mu = (x @ w1) + b1  # [1]
                    lnvar = (x @ w2) + b2  # [1]
                    var = lnvar.exp().clamp_min(1e-12)
                    std = var.sqrt()
                    mean_val = float(mu.item())
                    lb = float((mu - cal_factor * std).item())
                    ub = float((mu + cal_factor * std).item())
                return mean_val, lb, ub

            return predictor

        # ---------------------------
        # Sequestered evaluation split
        # ---------------------------

        eval_run_specs_df = sequestered_test_split.run_specs  # NOQA
        eval_scenario_state_df = sequestered_test_split.scenario_state

        metrics = ['expected_clip_score|test']

        # Collect all prompts and (model,metric) pairs needed for eval
        prompts = set()
        model_metrics = set()
        for _, row in eval_scenario_state_df.iterrows():
            prompt = row['scenario_state.request_states.request.prompt']
            model_name = row['scenario_state.adapter_spec.model']
            prompts.add(prompt)
            for metric in metrics:
                model_metrics.add((model_name, metric))

        prompts = sorted(list(prompts))
        prompt_embeddings = clip_embedding(prompts, batch_size=8)
        prompt_to_idx = {p: i for i, p in enumerate(prompts)}

        # Build predictors for each (model, metric)
        predictors = {
            (model_name, metric): craft_predictor(
                train_data_table,
                model_name,
                metric,
                banned_prompts=prompts,
            )
            for (model_name, metric) in model_metrics
        }

        # ---------------------------
        # Make predictions
        # ---------------------------

        predictions: list[InstancePrediction] = []
        metric = metrics[0]  # Only predict this metric for now

        for _, row in eval_scenario_state_df.iterrows():
            run_spec_name = row['run_spec.name']
            instance_predict_id = row['magnet.instance_predict_id']

            model_name = row['scenario_state.adapter_spec.model']
            prompt = row['scenario_state.request_states.request.prompt']

            # Get embedding of this prompt
            idx = prompt_to_idx.get(prompt, None)
            if idx is None:
                # Prompt not found in our precomputed list (shouldn't happen); skip
                a = 0 / 0

            emb = prompt_embeddings[idx]

            # Get predictor for (model_name, metric)
            predictor_fn = predictors.get((model_name, metric), None)
            if predictor_fn is None:
                # No predictor found; skip
                a = 0 / 0

            pred_mean, pred_lb, pred_ub = predictor_fn(emb)

            # NOTE: stat_name must match a HELM stat to be consumed downstream.
            # You can change "exact_match" to the one you care about, but the
            # framework right now only uses the mean.
            print(f'Prediction {prompt} {pred_mean} LB {pred_lb} UB {pred_ub}')
            predictions.append(
                InstancePrediction(
                    run_spec_name=run_spec_name,
                    instance_predict_id=instance_predict_id,
                    stat_name='expected_clip_score',  # or metric if that is what HELM expects
                    mean=float(pred_mean),
                )
            )

        return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Run example CLIP-based per-instance predictor'
    )

    parser.add_argument(
        'helm_suite_path',
        type=str,
        help="Path to HELM run outputs for a suite (usually '.../benchmark_output/runs/suite_name')",
    )

    args = parser.parse_args()

    predictor_instance = SRIInstancePredictor()
    predictor_instance(args.helm_suite_path)


if __name__ == '__main__':
    main()
