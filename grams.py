from SRToolkit.utils.grammar import MaxDepth
import numpy as np
import random

class MCTSNode:

    def __init__(self, state, parent=None, applied_rule=None, depth=0, prior=1.0):
        self.state = list(state)
        self.parent = parent
        self.applied_rule = applied_rule
        self.depth = depth
        self.children = []
        self.prior = prior
        self.visits = 0
        self.value = 0.0
        self.rule_path = []
        if parent is not None and applied_rule is not None:
            self.rule_path = parent.rule_path + [applied_rule]
            self.log_prob = parent.log_prob + np.log(max(float(applied_rule.weight), 1e-12))
        else:
            self.log_prob = 0.0
        self.applicable_rules = None
        self.exhausted = False 

class Grams:
    def __init__( self, grammar, reward, evaluator=None, error=1e-7, max_depth=100, c_param=1.4, use_puct=True, value_function="percentile"):
        self.grammar = grammar
        self.reward = reward
        self.evaluator = evaluator
        self.error = error
        self.max_depth = max_depth
        self.c_param = c_param
        self.use_puct = use_puct

        if value_function not in ["percentile", "minmax", "none"]:
            raise ValueError(f"Invalid value_function: {value_function}. Must be 'percentile', 'minmax', or 'none'.")
        self.value_function = value_function

        self.reward_min = float('inf')
        self.q_min = float('inf')
        self.q_max = float('-inf')
        self.q_observations = []  

        if evaluator is not None:
            self.n_points = evaluator.X.shape[0]
    
        # check that the probabilities of productions for each non-terminal sum to 1
        probs = {}
        for rule in grammar.to_dict()["rules"]:
            lhs = rule["lhs"]
            probs[lhs] = probs.get(lhs, 0) + rule["weight"]
        for lhs, total_p in probs.items():
            if not ((1.0 - 1e-12) < total_p < (1.0 + 1e-12)):
                raise ValueError(f"Productions for {lhs} do not sum to 1")

        grammar.add_constraint(MaxDepth(max_depth))


    def get_leftmost_nt(self, state):
        """ Return the index and symbol of the leftmost non-terminal in the state."""
        for idx, symbol in enumerate(state):
            if symbol in self.grammar.nonterminals:
                return idx, symbol
        return None, None

    def is_terminal(self, state):
        """ Check if the state is terminal (i.e., has no non-terminals)."""
        return self.get_leftmost_nt(state)[0] is None

    def _replay_derivation(self, rule_path):
        d = self.grammar.start_derivation(str(self.grammar.start))
        for rule in rule_path:
            d.apply(rule)
        return d

    def _positive_weight_rules(self, rules):
        positive_rules = []
        for rule in rules:
            weight = float(rule.weight)
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"Rule {rule} has invalid weight {rule.weight}")
            if weight > 0:
                positive_rules.append(rule)
        return positive_rules

    def _get_applicable(self, node):
        if node.applicable_rules is not None:
            return node.applicable_rules
        derivation = self._replay_derivation(node.rule_path)
        if derivation.complete:
            node.applicable_rules = []
        else:
            node.applicable_rules = self._positive_weight_rules(derivation.options())
        return node.applicable_rules


    def search(self, iterations, random_seed=None, k_rollouts=1):
        """ Perform MCTS search for a given number of iterations."""
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.history = []
        self.rule_history = []
        self.reward_min = float('inf')
        self.q_min = float('inf')
        self.q_max = float('-inf')
        self.q_observations = []   
        if self.evaluator is not None:
            self.min_rmse = float('inf')
            self.best_model = None

        root = MCTSNode([str(self.grammar.start)])

        for i in range(iterations):
            if self.evaluator is not None and self.evaluator.should_stop:
                break
            
            node = self.tree_policy(root)
            if node is None:
                break
                   
            rewards = []
            for _ in range(k_rollouts):
                if self.evaluator is not None and self.evaluator.should_stop:
                    break

                r = self.simulate(node)
                if r is not None:   
                    rewards.append(r)
                if self.evaluator is not None and self.min_rmse <= self.error:
                    break

            if not rewards:
                if self.evaluator is not None and self.evaluator.should_stop:
                    break

                self.backpropagate(node, self.failure_value())
                continue

            avg_reward = sum(rewards) / len(rewards)
            self.reward_min = min(self.reward_min, avg_reward)

            self.backpropagate(node, avg_reward)

            if self.evaluator is not None and self.min_rmse <= self.error:
                print(f"Stopping early at iteration {i} with RMSE {self.min_rmse}")
                return self.evaluator.get_results(top_k=1)

        if self.evaluator is not None:
            if not self.evaluator.models:
                return self.evaluator.get_results(top_k=0)
            results = self.evaluator.get_results(top_k=1)
            if len(results) == 0 or len(results[0].top_models) == 0:
                print("Warning: No valid models found during search.")
            return results

        return self.history

    def failure_value(self):
        """ Return a failure value for backpropagation when no valid reward is obtained."""
        # return self.reward_min if np.isfinite(self.reward_min) else 0.0
        if self.value_function == "none":
                    return self.reward_min if np.isfinite(self.reward_min) else 0.0
        return 0.0

    def score(self, parent, child):
        """ Compute the score for a child node based on UCT or PUCT formula.

        Q(s,a) se normalizira dinamično, z uporabo trenutnih statistik Q-vrednosti,
        opaženih kjerkoli v drevesu do tega trenutka:
        - "minmax":     min-max normalizacija (MuZero, Schrittwieser idr. 2020),
        - "percentile": normalizacija med 5. in 95. percentilom (Hafner idr. 2025),
        - "none":       brez normalizacije.
        """
        if child.visits == 0:
            q = 0.0
        else:
            q = child.value / child.visits
            if self.value_function == "minmax":
                if self.q_max > self.q_min:
                    q = (q - self.q_min) / (self.q_max - self.q_min)
                else:
                    q = 0.5
            elif self.value_function == "percentile":
                if len(self.q_observations) >= 2:
                    q05, q95 = np.percentile(self.q_observations, [5, 95])
                    if q95 > q05:
                        q = float(np.clip((q - q05) / (q95 - q05), 0.0, 1.0))
                    else:
                        q = 0.5
                else:
                    q = 0.5

        if self.use_puct:
            parent_visits = max(1, parent.visits)
            return q + self.c_param * child.prior * np.sqrt(parent_visits) / (1 + child.visits)

        if child.visits == 0:
            return float("inf")
        log_parent = np.log(parent.visits) if parent.visits > 0 else 0.0
        return q + self.c_param * np.sqrt(log_parent / child.visits)


    def tree_policy(self, node):
        """ Select a node to expand."""
        if self.is_terminal(node.state):
            if node.visits == 0:
                return node
            node.exhausted = True
            return None

        applicable = self._get_applicable(node)
        if not applicable:
            if node.visits == 0:
                return node
            node.exhausted = True
            return None

        if len(node.children) < len(applicable):
            return self.expand(node, applicable)

        candidates = [c for c in node.children if not c.exhausted]
        if not candidates:
            node.exhausted = True
            return None

        best = max(candidates, key=lambda c: self.score(node, c))
        result = self.tree_policy(best)
        if result is None: # already explored all children
            return self.tree_policy(node)   
        return result
  
    def expand(self, node, applicable):
        """ Expand a node by applying an untried rule."""
        tried = [c.applied_rule for c in node.children]
        untried = [prod for prod in applicable if prod not in tried]
        
        weights = [float(prod.weight) for prod in untried]
        prod = random.choices(untried, weights=weights, k=1)[0]
        
        total_applicable_weight = sum(float(rule.weight) for rule in applicable)
        if total_applicable_weight <= 0:
            raise ValueError("Applicable rule weights must sum to a positive value.")
        
        prior = float(prod.weight) / total_applicable_weight
        
        index, _ = self.get_leftmost_nt(node.state)
        new_state = list(node.state)
        new_state[index:index+1] = list(prod.rhs)

        new_node = MCTSNode(
            new_state,
            parent=node,
            applied_rule=prod,
            depth=node.depth + 1,
            prior=prior
        )
        node.children.append(new_node)
        return new_node

    def simulate(self, node):
        """ Simulate a rollout from the given node."""
        if self.evaluator is not None:
            return self._simulate_with_evaluator(node)
        else:
            return self._simulate_basic(node)

    def _simulate_basic(self, node):
        derivation = self._replay_derivation(node.rule_path)

        while not derivation.complete:
            rules = self._positive_weight_rules(derivation.options())
            
            if not rules:
                return 0.0
            
            weights = [float(rule.weight) for rule in rules]
            selected = random.choices(rules, weights=weights, k=1)[0]
            derivation.apply(selected)

        if not derivation.complete:
            return 0.0
        
        tokens = derivation.to_token_list()
        state_str = "".join(str(t) for t in tokens)
        applied_rules = derivation.to_parse_tree().productions_used()
        
        reward = self.reward(state_str)
        self.history.append((state_str, reward, None))
        self.rule_history.append(applied_rules)
        return reward

    def _simulate_with_evaluator(self, node):
        derivation = self._replay_derivation(node.rule_path)
        rollout_log_prob = node.log_prob

        while not derivation.complete:
            surviving_rules = self._positive_weight_rules(derivation.options())
            if not surviving_rules:
                return None

            weights = [float(r.weight) for r in surviving_rules]
            total = sum(weights)
            probs = [w / total for w in weights]

            choice_idx = np.random.choice(len(surviving_rules), p=probs)
            selected_rule = surviving_rules[choice_idx]
            rollout_log_prob += np.log(max(float(selected_rule.weight), 1e-12))
            derivation.apply(selected_rule)

        applied_rules = derivation.to_parse_tree().productions_used() 
        tokens = derivation.to_token_list()
        state_str = "".join(tokens)

        rmse = self.evaluator.evaluate_expr(tokens)
        if np.isnan(rmse) or np.isinf(rmse):
            return None
        if state_str not in self.evaluator.models:
            return None

        model_res = self.evaluator.models[state_str]
        num_params = len(getattr(model_res, 'parameters', []))

        current_reward = self.reward(
            rmse=rmse,
            k=num_params + 1,
            n=self.n_points,
            log_prob=rollout_log_prob
        )

        if rmse < self.min_rmse:
            self.min_rmse = rmse
            self.best_model = state_str

        self.history.append((state_str, current_reward, rmse))
        self.rule_history.append(applied_rules)
        return current_reward

    def backpropagate(self, node, value):
        """ Backpropagate the reward value up the tree."""
        while node is not None:
            node.visits += 1
            node.value += value
            if self.value_function in ("minmax", "percentile"):
                q = node.value / node.visits
                self.q_min = min(self.q_min, q)
                self.q_max = max(self.q_max, q)
                if self.value_function == "percentile":
                    self.q_observations.append(q)
            node = node.parent

def reward_function_bic(rmse, k,n,log_prob):
    rmse = max(rmse, 1e-12)
    log_posterior = - k/2 * np.log(n) + log_prob - n/2*np.log(2*np.pi) -n/2 - n*np.log(rmse)
    return log_posterior

def reward_function_rmse(rmse,**kwargs):
    rmse = max(rmse, 1e-12)
    return -rmse

def build_grammar(n_variables):
    variables = [f"X_{i}" for i in range(n_variables)]
    grammar = "E -> E '+' F [0.2] | E '-' F [0.2] | F [0.6]\n"
    grammar += "F -> F '*' T [0.2] | F '/' T [0.2] | T [0.6]\n"
    grammar += "T -> R [0.2] | V [0.4] | 'C' [0.4]\n"
    grammar += "R -> '(' E ')' [0.6] | 'sqrt' '(' E ')' [0.1] |'cos' '(' E ')' [0.1] | 'exp' '(' E ')' [0.1]| 'sin' '(' E ')' [0.1]\n"
    grammar += "\n".join([f"V -> '{v}' [{1/len(variables)}]" for v in variables])
    return grammar