import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from SRToolkit.utils.grammar import Grammar, MaxDepth
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.evaluation import SR_evaluator
from SRToolkit.dataset import Feynman
from SRToolkit.approaches import ProGED
from grams import Grams, reward_function_bic, build_default_grammar, reward_function_rmse

REWARD_FUNCTIONS = {
    "BIC": reward_function_bic,
    "RMSE": reward_function_rmse,
}


def run_one(name, seed, method_name, reward_name=None, max_evaluations=10000, k_rollouts=3,
            max_depth=20):
    benchmark = Feynman()
    dataset = benchmark.create_dataset(name, seed=seed)
    n_vars = dataset.X.shape[1]
    error_threshold = dataset.success_threshold 
    grammar_str = build_default_grammar(n_vars)   

    X_search, y_search = dataset.X, dataset.y

    grammar = Grammar.from_grammar_string(grammar_str, start="E")
    lib = SymbolLibrary.default_symbols(num_variables=n_vars)

    best_expr, expr_tokens, expr_params = None, None, None
    evaluation_calls = 0

    try:
        evaluator = SR_evaluator(X=X_search, y=y_search, symbol_library=lib,
                                 success_threshold=error_threshold, seed=seed,
                                 max_evaluations=max_evaluations)

        if method_name == "MC":
            grammar.add_constraint(MaxDepth(max_depth))
            proged_model = ProGED(grammar=grammar)
            proged_model.search(evaluator, seed=seed)

        else:
            reward_fn = REWARD_FUNCTIONS[reward_name]  
            grams = Grams(grammar=grammar, reward=reward_fn, evaluator=evaluator,
                         max_depth=max_depth,
                         use_puct=(method_name == "GRAMS-PUCT"), c_param=20,
                         value_function="percentile", error=error_threshold)
            grams.search(iterations=max_evaluations, random_seed=seed, k_rollouts=k_rollouts)

        results = evaluator.get_results(top_k=1)
        evaluation_calls = evaluator.total_evaluations
        if len(results) and len(results[0].top_models):
            r0 = results[0]
            min_rmse, success, best_expr = r0.min_error, r0.success, r0.best_expr
            expr_tokens = list(r0.top_models[0].expr)
            expr_params = list(r0.top_models[0].parameters)
        else:
            min_rmse, success = float("inf"), False

    except Exception as e:
        min_rmse, success = float("inf"), False
        print(f"[napaka] {name}/{seed}/{method_name}/{reward_name}: {e}")

    return {"equation_id": name, "seed": seed, "method": method_name,
            "reward_function": reward_name if method_name != "MC" else "N/A",   
            "min_error": min_rmse,
            "success": success,
            "best_expr": best_expr, "expr_tokens": expr_tokens,
            "expr_params": expr_params,
            "original_equation": dataset.original_equation if dataset.original_equation is not None else None,
            "ground_truth": dataset.ground_truth if dataset.ground_truth is not None else None,
            "n_variables": n_vars,
            "evaluation_calls": evaluation_calls}


RESULTS_PKL = "poskus2.pkl"
RESULTS_CSV = "poskus2.csv"
LIST_COLS = ["expr_tokens", "expr_params", "ground_truth"]
SAVE_EVERY = 10


def save_results(rows):
    df = pd.DataFrame(rows)
    df.to_pickle(RESULTS_PKL)
    df.drop(columns=LIST_COLS, errors="ignore").to_csv(RESULTS_CSV, index=False)
    return df


def load_done():
    if not os.path.exists(RESULTS_PKL):
        return [], set()
    df = pd.read_pickle(RESULTS_PKL)
    rows = df.to_dict("records")
    done = set(zip(df["equation_id"], df["seed"], df["method"], df["reward_function"]))
    return rows, done


def iter_results(tasks, n_jobs, max_evaluations=10000):
    try:
        yield from Parallel(n_jobs=n_jobs, verbose=10,
                            return_as="generator_unordered")(
            delayed(run_one)(name, seed, method, reward_name, max_evaluations=max_evaluations)
            for name, seed, method, reward_name in tasks)
    except TypeError:
        print("Starejši joblib: zapisujem po kosih namesto sproti.")
        chunk = max(1, n_jobs * 5)
        for i in range(0, len(tasks), chunk):
            yield from Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(run_one)(name, seed, method, reward_name, max_evaluations=max_evaluations)
                for name, seed, method, reward_name in tasks[i:i + chunk])


if __name__ == "__main__":
    benchmark = Feynman()
    dataset_names = benchmark.list_datasets(verbose=False)
    seeds = [0, 1, 2]
    grams_methods = ["GRAMS-UCT", "GRAMS-PUCT"]
    reward_names = ["BIC", "RMSE"]  
    
    N_JOBS = 7

    MAX_EVALUATIONS = 10000

    tasks = []
    for name in dataset_names:
        for seed in seeds:
            for method in grams_methods:
                for reward_name in reward_names:
                    tasks.append((name, seed, method, reward_name))
            tasks.append((name, seed, "MC", None))

    rows, done = load_done()
    if done:
        tasks = [t for t in tasks if t not in done]
        print(f"Nadaljujem: {len(done)} opravil je že izračunanih.")
    print(f"Skupaj opravil: {len(tasks)}")

    if tasks:
        for i, result in enumerate(iter_results(tasks, N_JOBS, MAX_EVALUATIONS), start=1):
            rows.append(result)
            if i % SAVE_EVERY == 0 or i == len(tasks):
                save_results(rows)

    df = save_results(rows)
    print(df.groupby(["method", "reward_function"])["success"].mean())
