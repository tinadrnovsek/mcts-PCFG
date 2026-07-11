# mcts-PCFG

Tina, komentarje bom pisal kar tukaj, pozneje lahko prestavimo drugam ali zbrišemo.

## Funkcija koristnosti za odkrivanje enačb, 11-Jul-2026

Naj bo $e$ matematični izraz, ki ga želimo ovrednotiti na podatkovni množici $D = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^n$. Če predpostavimo, da so napake na primerih iz $D$ medsebojno neodvisne in normalno porazdeljene okoli nič z nespremenljivo varianco, $\mathcal{N}(0, \sigma^2)$, potem lahko verjetje izraza za podano množico $D$ in vrednost variance $\sigma^2$ izračunamo z uporabo Bayesove formule:

$$
    \log P(e \mid D, \sigma) = 
        \log P(D \mid e, \sigma) + \log P(e) - \log P(D | \sigma).
$$

Zadnji člen $\log P(D | \sigma)$ je konstanten za vse izraze, ki jih vrednotimo na podani podatkovni množici $D$ in zato pri primerjalnem vrednotenju izrazov ne igra vloge. Verjetnost izraza $e$ lahko izpeljemo iz dejstva, da je njegova apriorna verjetnost enaka produktu verjetnosti vseh produkcijskih pravil iz drevesa izpeljave $t_e$ izraza $e$ v gramatiki $G$. Velja torej

$$ \log P(e) = \sum_{r \in t_e} \log P(r). $$

Verjetje podatkov, glede na predpostavko normalne porazdelitve lahko izračunamo z

$$
\begin{align}
    \log P(D \mid e, \sigma) 
    &= \sum_{i=1}^n \log \mathcal{N}(y_i;f_e(\boldsymbol{x}_i),\sigma^2) \\
    &= \dots \\
    &= - \frac{n}{2} \, \log (2 \pi \sigma^2) - \frac{\text{SSE}(e, D)}{2 \sigma^2},
\end{align}
$$

kjer je $SSE(e, D)$ vsota kvadratnih napak izraza $e$ na podatkovni množici $D$, ki jo izračunamo po formuli

$$
    SSE(e, D) = \sum_{i=1}^n (y_i - f_e(\boldsymbol{x}_i))^2.
$$

Ker je prvi člen izraza za logaritem verjetja $P(D \mid e, \sigma)$ konstanten za vse izraze ovrednotene na isti podatkovni množici, bi lahko funkcijo koristnosti za izraz $e$ na množici $D$ definiramo kot

$$
    U(e) = -\frac{\text{SSE}(e, D)}{2 \sigma^2} + \sum_{r \in t_e} \log P(r).
$$

Prvi člen funkcije koristnosti preferira izraze z nizko napako, drugi člen pa bolj verjetne izraze, ki so običajno tudi krajši.

Pri odkrivanju enačb obravnavamo izraze z neznanimi vrednostmi konstantnih parametrov $\boldsymbol{c}$. Algoritmi za odkrivanje enačb rešujejo numerični optimizacijski problem za iskanje optimalnih vrednosti teh parametrov na podani množici $D$

$$
    \boldsymbol{c}^* = \arg\min_{c} \text{SSE}(e, \boldsymbol{c}, D).
$$

Za potrebe implementacije učinkovite funkcije koristnosti lahko naredimo predpostavko $\text{SSE}(e, D) = \text{SSE}(e, \boldsymbol{c}^*, D)$. Ta predpostavka ni v skladu z Baysovim okvirjem, ki bi zahteval integriranje verjetja čez vse možne vrednosti $c$ parametrov izraza $e$

$$
    P(D \mid e, \sigma) = \int P(D \mid e, c, \sigma) \, p(c | e) \, dc.
$$

Integral bi lahko aproksimirali z Monte Carlo izračuni na vzorčenih vrednostih $c$ ali pa, bolje, z Laplaceovo aproksimacijo, ki naj ostane nadaljnje delo. Če te zelo mika, si lahko pomagaš z jezikovnimi modeli za hitro izpeljavo in implementacijo, a se mi to ne zdi nikakor nujno za dokončanje tvoje magistrske naloge.

Težava zgoraj definirane funkcije koristnosti je v tem, da je lahko njena vrednost poljubno velika in negativna. Poskus, da bi se temu izognili tako, da uporabimo fiksno monotono transformacijo $\phi$ (npr. sigmoidno ali tisto, ki sem jo priporočal zadnjič med sestankom in izhaja iz članka o odkrivanju enačb s spodbujevalnim učenjem) je slaba rešitev iz več razlogov. Najbolj očiten je ta, da rollout ocena, ki jo uporabljamo pri izbiri postane potem $\mathbb{E}[\phi(U(e))]$, kar pa ni enako kot $\phi(\mathbb{E}[U(e)])$. Torej ocenjena vrednost funkcije koristnosti ni več primerljiva z ekasktnimi izračuni te funkcije.

Zato normalizacijo izpeljemo tako, da vodimo statistike $Q_\text{min}$ in $Q_\text{max}$, glej na primer enačbo (5) na str. 12 članka [Schrittwieser in ost. 2020](https://arxiv.org/pdf/1911.08265). In z njimi koregiramo vrednost funkcije koristnosti preden izračunamo vrednosti (P)UCT. Nadaljnjo diskusijo težav z normalizacijo vrednosti funkcije koristnosti lahko prebereš v [Schmöcker in ost. 2025](https://arxiv.org/abs/2510.21275).

Po drugi strani, članek [Hafner in ost. 2025](https://www.nature.com/articles/s41586-025-08744-2) na str. 649 pove naslednje

> Normalizing by the smallest and largest observed returns would then scale returns down too much and may cause suboptimal convergence. To be robust to these outliers, we compute the range from the 5th to the 95th return percentile (Per) over the batch dimension...

Torej mi bi lahko uporabili podobno idejo za normalizacijo vrednosti funkcije koristnosti pred izračunom (P)UCT takole:

$$
    \overline{U}(e) = \min\left(\max\left(\frac{U(e) - U_{05}}{U_{95} - U_{05}},0\right),1\right),
$$

kjer $U_{05}$ in $U_{95}$ sta 5% in 95% kvantila opazovanih vrednosti funkcije koristnosti. Funkcija torej vsakič sproti normalizira vrednosti z uporabo bolj robustnih kvantilov in nato "zreže" njeno vrednost tako, da je na intervalu $[0,1]$.

Upam, da ti bodo ta navodila koristna pri nadaljevanju dela na odkrivanju enačb.


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
