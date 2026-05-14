# Pacman Capture-the-Flag: Agent Design Report

---

## 1. Introduction

Pacman Capture-the-Flag (CTF) is a multi-agent adversarial game built on the UC Berkeley Pacman framework. Two teams of two agents compete simultaneously on a symmetric maze divided down the middle. Each team's objective is to collect more than half of the opposing team's food pellets and deposit them safely on their own side, while simultaneously defending their own food supply from the opponent. An agent crossing the midline transitions from a ghost (which can eliminate enemy Pacmen) into a Pacman (which can collect enemy food but is vulnerable to the opposing ghosts). Power capsules exist in each half of the board — eating one causes all enemy ghosts to become temporarily "scared," reversing the predator/prey dynamic for a limited duration.

Crucially, agents do not have full information. When an enemy is beyond a certain sight range, its position is only observable through a noisy signal: the observed position may deviate from the true position by up to ±2 cells along each axis. This partial observability requires agents to reason under uncertainty about where opponents are.

The solution presented in this report is a role-specialised two-agent team implemented in `myTeam.py`. One agent, `GreedyPointAgent`, is dedicated to offense: it collects the opponent's food as efficiently as possible while using adversarial reasoning to escape when threatened. The second agent, `InterceptDefenseAgent`, is dedicated to defense: it patrols structurally significant positions on the home side and actively intercepts invaders when they are detected.

The key techniques underlying this solution are: danger-aware A* search for pathfinding; minimax with alpha-beta pruning for adversarial escape decisions; probabilistic belief tracking for enemy localization under partial observability; and graph-based choke-point analysis for deriving high-value patrol targets. Together, these techniques produced a team that achieved joint first place in the preliminary tournament evaluation with 13 wins, 0 ties, and 1 loss.

---

## 2. Theoretical Foundation

### 2.1 Heuristic Search: A*

A* is a best-first graph search algorithm that finds the least-cost path from a start node to a goal. At each step it expands the node with the lowest value of `f(n) = g(n) + h(n)`, where `g(n)` is the actual cost accumulated to reach node `n` from the start, and `h(n)` is a heuristic estimate of the remaining cost to the goal. When the heuristic is *admissible* — meaning it never overestimates the true remaining cost — A* is guaranteed to find an optimal (least-cost) path.

A* generalises Dijkstra's algorithm, which uses `h(n) = 0`. The addition of a well-chosen heuristic allows A* to focus its search toward the goal, examining fewer nodes in practice while preserving optimality guarantees.

In this project, the search is run over the maze grid rather than a game-tree, making `g(n)` the sum of step costs along a path of cells. The key insight is that step costs need not be uniform: by setting `cost(cell) = 1 + danger(cell)`, where `danger(cell)` is a function that assigns higher values to cells near enemy ghosts, the same A* algorithm that solves shortest-path problems also solves the joint problem of finding a short path that avoids dangerous areas. This eliminates the need for a separate filtering stage.

A notable strength of A* is its flexibility — changing the cost function or goal predicate entirely changes the agent's behaviour without requiring any changes to the search loop itself. A limitation in real-time settings is that the path is computed fresh each turn from the current game state, so prior computation is discarded. In environments where the cost field changes rapidly (e.g., due to fast-moving ghosts), this is not a significant issue, but it does mean the algorithm cannot amortise computation across turns.

### 2.2 Adversarial Search: Minimax with Alpha-Beta Pruning

Many AI problems involve an agent acting in an environment where another agent is actively working against it. Minimax is the foundational algorithm for reasoning in such two-player, zero-sum games. It models the problem as a game tree: at nodes where the agent to move is the maximiser (our agent), we choose the action leading to the highest-valued child; at nodes where the opponent moves (the minimiser), we assume they will choose the action leading to the lowest-valued child. The value of a leaf node is given by a heuristic evaluation function. Values are backed up recursively to the root, and the maximiser picks the root action corresponding to the highest-valued subtree.

A standard minimax tree grows exponentially with depth. Alpha-beta pruning addresses this by maintaining a window `[alpha, beta]` — the best score the maximiser is already guaranteed and the best score the minimiser is already guaranteed, respectively. When a subtree is provably outside this window (it cannot affect the backed-up root value), it is pruned without being searched. In the worst case alpha-beta provides no benefit, but in practice it typically halves the effective search depth, allowing roughly twice as deep a search in the same time budget.

The depth of search (measured in *plies*, where one ply is one agent's move) directly controls the lookahead horizon. A depth-4 minimax in a 1v1 pursuit game means our agent can reason about sequences of four alternating moves (four of ours, four of the ghost's), which is enough to evaluate short-term escape corridors and dead-ends.

A strength of minimax is that it explicitly reasons about the adversary's best responses rather than assuming the opponent is passive. The principal limitation here is the *assumption of optimality*: a real opponent may play suboptimally, but minimax always expects the worst. Additionally, minimax scales poorly with the number of agents: a full three-agent tree (us plus two ghosts) has branching factor cubed per ply, making it impractical at meaningful depth without further approximations such as expectimax.

### 2.3 Probabilistic Belief Tracking

When an enemy agent is outside the sight range, the game provides a noisy positional observation rather than the true position. Reasoning effectively under this partial observability requires maintaining and updating a *belief* — a representation of what positions the enemy is plausibly occupying given all past observations.

A belief can be represented as a probability distribution over all grid cells, or more compactly, as the *support set* of cells that are consistent with observations so far (i.e., a set of possible positions without explicit probabilities). The support-set approach is used here.

Belief maintenance proceeds each turn in two steps: a *prediction* (or *transition*) step and an *update* (or *observation*) step. In the prediction step, we account for the fact that the enemy may have moved since the last observation: each believed position is expanded to include all cells reachable in one move (its legal neighbours), since the enemy could have moved to any of them. In the observation step, we incorporate the new noisy reading by intersecting the expanded set with the set of positions consistent with the latest observation. For this game's noise model — a scrambled position within ±2 cells along each axis — the consistent set is a 5×5 box (minus walls) centred on the observed position.

A useful refinement is *multi-agent synchronisation*: if two allied agents both maintain beliefs about the same enemy, the intersection of their individual beliefs (each constrained by its own vantage point) is strictly no larger than either belief alone and may be substantially tighter.

The support-set approach is computationally simple — intersection and BFS diffusion are both O(|belief|) — and practically sufficient when the belief remains tight. Its main limitation is the assumption of *uniform* diffusion: every reachable cell is treated as equally likely. This is a poor model for an enemy that is following a specific strategy. When the enemy moves purposefully in one direction, the true position lags behind the diffused belief, causing the belief to "spread out" faster than it should.

### 2.4 Structural Analysis: Entry Points and Choke Detection

A choke point (or bottleneck) in a graph is a node whose removal significantly increases the shortest path distance between two other nodes. In a maze, choke points correspond to narrow corridors or junctions through which most paths must pass. Defending choke points is inherently more efficient than defending the entire boundary: a single agent positioned at a choke point can cut off access to a large cluster of food pellets.

Computing choke points rigorously (e.g., finding minimum vertex cuts) is expensive for arbitrary graphs. A practical approximation is used here: for each food cluster on the home side, find the shortest path from the midline boundary to the cluster's representative (medoid) food. Walk this path cell by cell; for each cell, run a second BFS with that cell blocked. If the detour is large enough (at least `MIN_CHOKE_GAIN = 3` steps longer), the cell is a choke point worth guarding. Otherwise, fall back to a cell near the start of the path as a default patrol target.

Food clustering is done via single-link clustering using a union-find (disjoint-set) structure: two food items that are within a threshold maze distance are merged into the same cluster. The cluster medoid — the food item that minimises total distance to all other items in the cluster — serves as the cluster representative for the path-finding step.

The strength of this approach is that it is entirely structural: it depends only on the maze topology and the initial food layout, not on dynamic game state. This means it can be precomputed once at the start of the game. The limitation is precisely this static nature — as food is eaten during the game, the original cluster representatives and choke points may no longer reflect the current food distribution, potentially leaving the defender guarding positions that no longer protect valuable food.

---

## 3. Final Agent Description

### 3.1 Team Overview

The team consists of two agents with clearly separated roles. `GreedyPointAgent` is the offensive agent, responsible for crossing the midline, collecting the opponent's food, and returning it safely. `InterceptDefenseAgent` is the defensive agent, responsible for preventing enemy Pacmen from collecting food on the home side.

The decision to specialise roles is motivated by conflicting optimisation objectives. An agent optimising for offense needs to be positioned deep in enemy territory, which makes it unavailable to intercept invaders on the home side. Conversely, an agent stationed near home entry points cannot simultaneously be collecting remote food. Running one instance of a single general-purpose agent per side would likely result in both agents doing neither job well. The two-agent specialisation allows each agent to apply techniques tailored precisely to its role.

### 3.2 Offensive Agent: GreedyPointAgent

At a high level, `GreedyPointAgent` operates as a hierarchical decision tree: each turn it classifies its current situation into one of several cases and applies the most appropriate strategy to each case.

**Main decision loop.** The entry point is `chooseAction()` (`myTeam.py:265`). At the start of each turn, the agent updates its enemy belief trackers and computes three key facts: `in_danger` (whether a non-scared ghost is within `DANGER_RADIUS = 4` maze cells), `carrying` (how many food pellets are being carried), and `must_return` (whether the agent should head home based on carrying threshold, remaining food, or time pressure).

The two primary branches are:

- **Danger branch** (`myTeam.py:299–356`): If `in_danger` is true, the agent clears any committed food target and enters a defensive sub-routine. It first checks whether a power capsule is within reach (`CAPSULE_RUSH_RADIUS = 7`) and closer than home — if so, it uses minimax to navigate toward the capsule. Otherwise, if it is carrying enough food or otherwise must return, it uses minimax to navigate home. If neither applies, it checks whether any nearby food is safe to eat before defaulting to a minimax-driven retreat.

- **Safe travel branch** (`myTeam.py:358–386`): If not in danger and the agent must return home, it uses A* to navigate home with a danger cost overlay. Otherwise, it selects a food target and uses A* to navigate to it.

**A* with danger costs.** `_aStarFirstAction()` (`myTeam.py:520`) implements Dijkstra (h=0) over the maze grid. The step cost is `1.0 + danger_fn(next_cell)`, where `danger_fn` is built by `_buildDangerFn()` (`myTeam.py:478`). The danger function assigns additive costs to cells near known ghost positions: a cell at distance 0 adds `weight * 5`, distance 1 adds `weight * 2`, distance 2 adds `weight * 0.6`, distance 3 adds `weight * 0.2`. This cost gradient smoothly steers the A* path away from ghosts without hard-coded exclusion zones. When searching for food targets, a weight of 4.0 is used; when searching for home, a weight of 10.0 would be used implicitly through the minimax fallback.

Only confirmed ghost positions and *tight* beliefs (belief set size ≤ `BELIEF_DIFFUSE_LIMIT = 50`) contribute to the danger function. This design choice prevents diffuse beliefs — which are spread over much of the map — from uniformly penalising all cells and causing the A* path to deteriorate into an uninformative walk.

**Minimax escape.** `_minimaxEscape()` (`myTeam.py:563`) runs a 1v1 minimax tree (4 plies, i.e., 8 half-plies with alpha-beta) against the nearest identified ghost. The evaluation function at leaf nodes is `score = -goal_dist * 10 + min(ghost_dist, 6) * 4`, with bonuses for reaching home (+50) or a capsule (+200). The heavy penalty on `goal_dist` keeps the agent moving toward its objective; the ghost-distance term rewards maintaining separation. Capture (ghost_dist = 0) is scored at `−10^6 − 1000 * carrying` to represent the catastrophic loss of all carried food. Minimax is used exclusively in danger scenarios, not during normal food collection, to keep the per-turn computation budget predictable.

**Target commitment.** `_selectFoodTarget()` (`myTeam.py:390`) implements a target tracker to avoid flip-flopping between equidistant food items, a common failure mode for purely greedy agents in open layouts. Once a food target is selected, the agent commits to it across turns. If the agent makes no progress toward the target for `STALE_THRESHOLD = 5` consecutive turns, the target is blacklisted for `BLACKLIST_TTL = 30` turns and a new target is chosen. Target scoring balances proximity with an ambiguity penalty (`score = dist + 0.5 * ambiguity`) where ambiguity counts how many other food items are within Manhattan distance 2 — highly clustered targets tend to cause future flip-flop even after one is selected.

**Safe food evaluation.** When in danger, `_safestFood()` (`myTeam.py:691`) filters nearby food by safety margin: a food item at position `f` is considered safe only if `ghost_dist(f) - my_dist(f) >= SAFE_FOOD_MARGIN`. With `SAFE_FOOD_MARGIN = 1`, this is a relatively permissive criterion (deliberately loosened from an earlier value of 2) to prevent the agent from over-refusing food in open spaces.

### 3.3 Defensive Agent: InterceptDefenseAgent

`InterceptDefenseAgent` operates as an explicit three-state machine: **PATROL**, **INTERCEPT**, and **RETREAT**. The state is selected freshly each turn based on whether invaders are detected and whether the agent is currently scared.

**State selection.** In `chooseAction()` (`myTeam.py:1144`), the agent first updates beliefs, detects newly eaten food, and localises any Pacman invaders. If the agent is scared and invaders are present, it enters RETREAT; if invaders are present (and the agent is not scared), it enters INTERCEPT; otherwise it patrols.

**Invader localisation.** `_localizeInvaders()` (`myTeam.py:1181`) uses a three-tier priority: (1) exact position if the invader is within sight range; (2) the most recently eaten food cell if food has disappeared within the last `EATEN_FRESH_TURNS = 6` turns; (3) the closest home-side belief cell if the belief set is tight enough. The eaten-food heuristic is particularly valuable because food disappearance is a reliable indirect signal of enemy position even when the enemy is out of sight.

**INTERCEPT state.** `_intercept()` (`myTeam.py:1212`) picks the best entry point for the defender to race to using `_bestInterceptCell()` (`myTeam.py:1237`). For each candidate entry, a *race score* is computed: `score = race_margin * 2 + on_path_bonus + food_value * 0.5`, where `race_margin = inv_to_cell - me_to_cell`. Entries where the invader would arrive more than 2 steps ahead are filtered out (the race is lost). This scoring favours entries the defender can reach at least as fast as the invader and that are structurally close to the invader's path home. If the defender is already at the best intercept cell, it directly chases the invader.

An anti-bait mechanism in `_shouldChase()` (`myTeam.py:1380`) prevents the defender from being drawn out of position by a shallow incursion. An invader near the boundary with no recently eaten food is classified as potential bait and ignored unless the defender is already very close.

**PATROL state.** `_patrol()` (`myTeam.py:1297`) cycles through the precomputed entry points sorted by y-coordinate, dwelling at each for `DWELL_TURNS = 4` turns before advancing to the next. `_currentPatrolTarget()` (`myTeam.py:1325`) can override this cycle: if food has been recently eaten, the defender diverts to the entry point closest to the eaten cell, making patrol responsive to live game events. If no entry points were found during initialisation (e.g., on an unusually open map), the patrol falls back to a thinned subset of home boundary cells.

**RETREAT state.** `_retreat()` (`myTeam.py:1266`) scores each legal action by a combination of distance maintained from the invader and proximity to the nearest patrol entry. The scoring keeps the agent in the home half, rewards distance from the invader, and penalises stopping, ensuring the agent remains active while waiting for its scared timer to expire.

### 3.4 Strengths and Weaknesses

**Strengths.** The minimax escape gives the offensive agent genuine multi-step reasoning during pursuits, which is qualitatively better than one-step lookahead and largely explains its ability to navigate narrow corridors under pressure. The entry-point patrol is structurally grounded rather than arbitrary, meaning the defender naturally covers high-value interception positions without needing to discover them through trial and error. Belief tracking keeps both agents aware of enemy positions even during periods of prolonged invisibility. Target commitment avoids the erratic movement patterns that afflict purely greedy food collectors in open layouts.

**Weaknesses.** The minimax tree models only one ghost at a time — the nearest threat. In scenarios where two enemy ghosts coordinate a pincer approach, the agent may escape one ghost only to collide with the other. Entry-point analysis is static and computed from the initial food layout; as food is eaten during the game, the structural rationale for certain patrol positions erodes but the defender does not adapt. The belief model uses uniform diffusion, which does not capture the fact that the enemy is likely following a deliberate strategy; in open maps this causes the belief set to grow large and become uninformative quickly. The two agents do not share tactical information beyond the belief intersection — the offense agent does not signal to the defense that it is carrying a full load and needs an open escape corridor, and the defense does not inform offense of known enemy positions beyond what is already visible.

---

## 4. Observations

### 4.1 Tournament Performance

In the preliminary tournament evaluation, the team achieved joint first place with a record of 13 wins, 0 ties, and 1 loss across 14 matches, winning 93 out of approximately 140 individual maps played. The zero-tie count is notable: every match was decided, suggesting the team does not get locked into inconclusive endgames where neither side can collect the remaining food. This is consistent with the offense agent's time-pressure logic, which forces a return home when `time_left < home_dist + 20`, ensuring food is deposited rather than carried until the clock expires.

### 4.2 Offensive Effectiveness

Winning 93 maps out of 140 (~66%) points to consistently effective offense. The combination of the danger-cost A* (which routes around ghost patrol zones) and minimax escape (which handles active pursuit) appears to produce an agent that collects food efficiently under normal conditions and survives chases in most cases. The `CARRY_THRESHOLD = 5` setting also plays a role: rather than hoarding food indefinitely, the agent returns home after collecting 5 pellets, banking points frequently and limiting the potential loss from a single capture.

### 4.3 The Single Match Loss

The one match loss (and 37 individual map losses spread across all 14 matches) is worth examining. The most plausible explanations are: (a) an opponent whose two ghosts coordinated such that the offense agent was forced into a dead-end by one ghost while the second covered the exit — a scenario the 1v1 minimax cannot resolve correctly; or (b) an opponent that used power capsules aggressively, making both enemy ghosts scared for long windows and stripping the defense agent of its ability to engage invaders during those periods. Without access to the specific match replays, these remain hypotheses rather than confirmed root causes.

### 4.4 Defensive Observations

The structural entry-point patrol worked well on tournament maps, which tend to feature corridor-heavy layouts with natural choke points. On layouts that are more open, the patrol falls back to boundary patrol, which is less targeted but still provides coverage. One edge case observed during local testing involves the anti-bait check in `_shouldChase()`: an invader that loiters just inside the boundary without eating any food is correctly identified as potential bait and ignored. However, a sufficiently patient opponent can exploit this by making small incremental incursions — each too shallow to trigger a chase — and gradually collecting boundary food without ever triggering the INTERCEPT state.

### 4.5 Belief Tracking in Practice

The belief tracking performed reliably in most scenarios, particularly when the two allied agents were separated (as they typically are, with one in enemy territory and one at home). The multi-agent belief intersection was effective in tightening estimates when the offense agent was on the far side of the map and the defense agent was near home. The `BELIEF_DIFFUSE_LIMIT = 50` cutoff prevented false threat signals from large belief sets, but on very open maps the belief could exceed this limit quickly, effectively blinding the agent to threats it could not directly observe.

---

## 5. Recommendations and Conclusion

### 5.1 Recap

This report has described a two-agent Pacman CTF team that separates offensive and defensive responsibilities across two specialised agents. The offensive agent uses A* with a danger cost overlay for routine food collection and switches to a depth-4 minimax tree with alpha-beta pruning when threatened by an enemy ghost. The defensive agent uses precomputed choke-point entry positions as patrol targets and engages a three-state machine (PATROL / INTERCEPT / RETREAT) driven by probabilistic belief tracking and eaten-food signals. The team placed joint first in the preliminary tournament evaluation.

### 5.2 Improvement Areas

**Multi-ghost adversarial search.** The most structurally significant gap is the 1v1 minimax assumption. Extending `_minimaxEscape()` to a two-ghost adversarial tree — or using expectimax with both ghosts as chance nodes — would give the offense agent meaningful reasoning about being double-teamed. The branching factor increase is manageable at moderate depth if alpha-beta is applied carefully.

**Dynamic entry-point updates.** The `EntryPoints` analysis is computed once at game start and never updated. A straightforward improvement would be to recompute entry points when a significant fraction of the home-side food has been eaten, or at least to invalidate entries whose protected food cluster has been fully consumed. This would keep the patrol meaningfully targeted in the late game.

**Intent-aware belief diffusion.** Replacing the uniform diffusion model with one that weights moves toward enemy food more heavily would keep the belief set tighter in open layouts. Even a simple directional bias — spreading probability mass preferentially toward the enemy's visible food cluster — would improve threat estimates in scenarios where the current model becomes unhelpfully diffuse.

**Offensive-defensive coordination.** Currently, the two agents share only belief intersection data. Explicit tactical signals — for example, the offense agent flagging that it is heavily loaded and within a few steps of home — could allow the defense agent to temporarily move toward the boundary to create a safe deposit corridor. This kind of simple teammate signalling could meaningfully reduce cases where the offense agent is caught while carrying a large payload near home.

**Parameter optimisation.** The agent has numerous hand-tuned constants (`CARRY_THRESHOLD`, `DANGER_RADIUS`, `MINIMAX_DEPTH`, `SAFE_FOOD_MARGIN`, `DWELL_TURNS`, and others). A systematic parameter search — even a coarse grid search over a held-out set of tournament layouts — could find better-performing combinations than the manually selected values. A more sophisticated approach using reinforcement learning or evolutionary strategies could jointly optimise parameters that interact with each other.

**Comparison with MCTS.** An alternative MCTS-based implementation exists in `diepTeam.py`. MCTS with the UCB1 selection criterion is an *anytime* algorithm — it improves its estimate with each additional simulation, making it naturally adaptive to variable per-turn time budgets. A direct head-to-head comparison on standardised layouts would clarify whether the fixed-depth minimax or MCTS produces better decisions in the time window available per turn in this game.

### 5.3 Conclusion

The solution demonstrates that role specialisation, combined with principled algorithmic choices tailored to each role, can produce a highly competitive team for the Pacman CTF domain. The offensive agent's layered decision hierarchy — greedy A* in safe conditions, minimax escape under threat — and the defensive agent's structurally-motivated patrol strategy together produce behaviour that is robust across diverse layouts and opponents. The primary directions for improvement are extending adversarial reasoning to multi-ghost scenarios and making the defensive patrol adaptive to the evolving food landscape.
