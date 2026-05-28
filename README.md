# mcts-PCFG

Tina, komentarje bom pisal kar tukaj, pozneje lahko prestavimo drugam ali zbrišemo.

## Pregledovanje kode, 25-maj-2026

Zaenkrat sem pozorno pregledal kodo za razred `mcts` in predlagam nekaj popravkov. Najbrž bi morala pred nadaljevanjem podobno popraviti kodo za odkrivanje enačb, nato pa se lotiva iskanja ustrezne funkcije koristnosti (nagrade).

1. V tretjem poglavju članka [(Czech in ost. 2020)](https://arxiv.org/pdf/2012.11045), ki sem ti ga poslal pred časom, predlagajo algoritem PUCT (slednjega pravzaprav predlagajo drugi članki, ki jih ta članek citira v drugem odstavku tretjega poglavja, a tukaj se mi je zdela razlaga bolj jasna). PUCT je za nas bolj ustrezen, ker pri izbiri pravil lahko upoštevamo njihove apriorne verjetnosti, kot jih definira gramatika.

```{python}

# PUCT implementation, see https://arxiv.org/pdf/2012.11045, section 3
# This is more suitable for our setting where we have a prior probability from the PCFG
def best_child(self, c_param=1.4):
  parent_visits = max(1, self.visits)

  def puct(child):
    q = 0.0 if child.visits == 0 else child.total_value / child.visits
    prior = child.applied_rule.prob()
    exploration = c_param * prior * np.sqrt(parent_visits) / (1 + child.visits)
    return q + exploration

  return max(self.children, key=puct)
```

Pozor: med popravljanjem kode sem `value` zamenjal s `total_value` (`self.total_value = 0.0`).

2. V `expand` bi morali upoštevali apriorne verjetnosti pravil pri njihovi izbiri. Trenutna verzija kode namreč upošteva vrstni red pravil (in ne njihovih verjetnosti), zato predlagam naslednjo spremembo metode `expand`, takoj za `tried = ...`:

```{python}

untried = [prod for prod in applicable if prod not in tried]
weights = [prod.prob() for prod in untried]
prod = random.choices(untried, weights=weights, k=1)[0]

new_state = list(node.state)
new_state[index:index+1] = list(prod.rhs())

new_node = MCTSNode(new_state, parent=node, applied_rule=prod, depth=node.depth + 1)
node.children.append(new_node)
return new_node
```

3. Sprotno računanje pravil `applicable = [r for r in self.rules if r.lhs() == nt]` se ponavlja prevečkrat, zato je morebiti bolje, če zgradiš slovar `rules_by_lhs`

```{python}
self.rules_by_lhs = {}
for rule in self.grammar.productions():
    self.rules_by_lhs.setdefault(rule.lhs(), []).append(rule)
```

in ga nato ga uporabljaš takole `applicable = self.rules_by_lhs.get(nt, [])`.

4. V `simulate` lahko bolj podrobno poročaš o morebitnem neuspehu takole:

```{python}
rules = self.rules_by_lhs.get(nt, [])
if not rules:
    raise ValueError(f"No production rules available for nonterminal {nt}")
probs = [r.prob() for r in rules]
selected = random.choices(rules, weights=probs, k=1)[0]
```

5. Nisem čisto prepričan, a se mi zdi, da na začetku `search` manjka `self.rollout_results = []`. Ker namreč, ob večkratnih zaporednih klicih bi najbrž morali ta seznam sprazniti?


## Posplošitev funkcije koristnosti

Ugotavljam, da je Ginijev indeks preveč enostavna funkcija koristnosti, ker doseže maksimum za zelo kratko besedo (ki jo hitro lahko zadane tudi vzorčenje Monte Carlo). Zato predlagam, da zamenjamo funkcijo koristnosti tako, da bo njena maksimalna vrednost odvisna tudi od dolžine besede.

Definirajmo najprej Ginijev indeks za podano besedo $w$

$$\text{Gini}(w) = 1 - \sum_{a \in \Sigma} \left( \frac{|w|_a}{|w|} \right)^2, $$

kjer je $|w|_a$ število znakov $a$ v besedi $w$, $|w|$ pa njena dolžina. Ker je maksimalna vrednost Ginijevega indeksa enaka

$$1 - \frac{1}{|\Sigma|},$$

lahko Ginijev indeks normaliziramo tako, da ima vrednost na intervalu $[0,1]$:

$$\text{Gini}_N(w) = \frac{\text{Gini}(w)}{1 - \frac{1}{|\Sigma|}}. $$

Zdaj lahko definiramo funkcijo koristnosti, ki upošteva dolžino besede, takole:

$$U(w) = \text{Gini}_N(w)^\gamma |w|^\alpha,$$

kjer sta $\alpha$ in $\gamma$ parametra, ki nam omogočajo uravnavati moč vpliva Ginijevega indeksa in dolžine besede na njeno koristnost. Nastavitev $\alpha = 0$ in $\gamma = 1$ nam omogoča iskanje besede z največjim Ginijevim indeksom. Večje vrednosti $\alpha$ povečajo vpliv dolžine besede, večje vrednosti $\gamma$ določajo kako izrazit je vpliv Ginijevega indeksa. Lahko preizkusimo različne nastavitve, jaz sem se igral z $\alpha = 0.5$ in $\gamma = 10$.
