# mcts-PCFG

Drevesno preiskovanje Monte Carlo (MCTS) izrazov, generiranih z verjetnostno kontekstno-neodvisnimi gramatikami (PCFG)

- **`grams.py`** —  vsebuje razreda `MCTSNode` in `Grams`, funkciji koristnosti (`reward_function_bic`, `reward_function_rmse`) in `build_grammar`.
- **`run_feynman.py`** — odkrivanje 100 Feynmanovih enačb z `Grams` in vzorčenjem Monte Carlo (ProGED), rezultate pa shrani v `.pkl`/`.csv`.
- **`mcts.ipynb`** — zvezek s poskusi in vizualizacijo za več gramatik, izdelan za magistrsko delo