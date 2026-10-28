# GravitrueX AI Project Notebook — Architecture, Training, Memory
**Updated:** 2026-10-28  
**Status:** Research notes and proposed experiments; not yet implemented unless explicitly stated below.

## 1. Current baseline and immediate priority

Current large-model run reported a 2048 × 5 residual network (~17.04 million parameters), hybrid CPU+GPU collection, and only a few PPO updates in the cited diagnostics. The reported recent-combat gap was large (blue hit rate roughly 3.7% vs red 30.5%, blue damage about 37 vs red about 337.5 over the last 20 battles). This is a warning signal, not a verdict on convergence: three PPO updates are far too few to establish what the model can ultimately learn. It does point to aim/positioning/target selection, observation/reward/teacher quality, and evaluation methodology as priorities—not simply adding more parameters.

**First priority: trustworthy measurements and stable training.**
- Repair/verify timing and memory telemetry (wall-clock time, CPU time, GPU time, CUDA allocated vs reserved vs peak vs free memory). Do not optimize a metric until its meaning is reliable.
- Compare GPU-only against hybrid collection using the same seeds/configuration. In the reported run, the CPU side contributed only a small fraction of GPU records while using host RAM; keep it only if it adds enough useful data to justify its cost.
- Benchmark 768×3, 1536×4, and 2048×5 (or a similar controlled ladder) using multiple seeds and fixed opponent pools. Track win rate, hit rate, damage dealt/taken, survival, entropy, approximate KL, clip fraction, value explained variance, records/sec, wall-clock training time, host RAM and VRAM.
- Save each baseline/config/checkpoint and compare matched seeds; avoid changing architecture, rewards, teacher, and optimizer simultaneously.

## 2. Learning ideas already identified

### PPO baseline and evaluation
Keep PPO as the baseline while its implementation and diagnostics are made reliable. It is a reasonable on-policy method for the current factorized discrete movement/fire actions. Do not switch to DQN/Rainbow or SAC simply because they exist:
- DQN/Rainbow would require an action-space/replay refactor; off-policy replay can improve sample reuse but adds instability and complexity in self-play.
- SAC/TD3 are more naturally suited to continuous controls; the current controls are mostly categorical.
- AlphaZero-style search is not the first fit for fast real-time combat without a strong model/search interface.

### Rewards, GAE, and teacher data
- Audit positive teacher-fire examples and distinguish “policy fired” from “the shot was useful.” Sparse positive teacher labels can make imitation ineffective.
- Run controlled A/B tests for current team reward versus individual reward and for a small number of gamma/lambda settings. Do not assume increasing gamma is automatically better: long horizons depend on critic quality, reward density, and reliable credit assignment.
- Add or alter reward terms only with a defined hypothesis and a regression test; dense shaping can accidentally teach easy-to-measure behaviours rather than winning.
- Record outcome distributions, not just a single rolling average; use fixed evaluation seeds and report confidence/variance across runs.

### Opponent diversity and strategic learning
- Build a frozen checkpoint/opponent pool: older snapshots, varied heuristics, and self-play opponents. Periodically evaluate against the pool, not only the latest self.
- Start with controlled matchups and comparable compositions, then expand to asymmetric team compositions.
- Consider a centralized-training/decentralized-execution (CTDE) critic: during training, a critic can observe global/team state while deployed ship actors remain local. This is more specifically a MAPPO-like improvement than merely having an independent critic.
- Later, consider entity-based attention or Deep Sets so the policy can reason over a variable list of ships/projectiles rather than only fixed “nearest entity” slots. Recurrent memory should be added only where tests show the present observation is insufficient.

## 3. Should the agent be split into several networks?

**Likely yes in some form, but start with modular outputs and shared features rather than many fully independent networks.** Several different techniques are often conflated:

1. **Shared backbone + specialized heads.** One observation encoder learns common features; movement, aiming, firing, target choice and strategic intent get separate output heads. This is the cheapest, easiest-to-train first experiment, and resembles the semantically meaningful action heads used by OpenAI Five.
2. **Hierarchical reinforcement learning / options.** A high-level manager selects a goal or temporally extended skill (“focus target”, “disengage”, “cover ally”, “maintain range”); a low-level policy controls turning, thrust, and firing. Each option has initiation and termination conditions and runs for a variable duration.
3. **Mixture of Experts (MoE).** Several expert sub-networks specialize, while a gating/router selects or combines a small subset for each situation. This is conditional computation, not necessarily several independent agents. Routing imbalance and training complexity are real concerns; large MoE results in language models do not guarantee an MoE will help a small game simulator.
4. **Separate specialist policies.** An aim/fire policy, movement policy and tactical planner can be trained and replaced separately, but they need stable inputs/outputs, arbitration rules, and a way to measure the team objective. They may conflict (movement wants to retreat while aim wants to keep firing), and independently optimizing local metrics can reduce overall win rate.
5. **Multiple teammates with a team-level coordinator.** A commander issues goals to local ship policies. This can help team tactics, but communication and coordination need to be represented/trained; it is not enough to add a “commander” label.

### Recommended GravitrueX experiment ladder
- **M0 — baseline:** save a stable, evaluated checkpoint plus config and fixed test seeds.
- **M1 — low-risk modularity:** preserve shared encoder and PPO update; separate/clean up heads for movement, aim/target choice, and fire/weapon timing. Add per-head diagnostics. Keep a joint team-level win reward.
- **M2 — explicit intent:** add a small tactical head that chooses among a few goals (focus, kite/retreat, protect/escort, regroup/formation, attack opportunistically). Hold the low-level controller unchanged initially. Goal changes should have hysteresis/cooldowns to avoid rapid oscillation.
- **M3 — temporal options:** let the high-level goal persist for a limited horizon or until termination conditions are met. Compare against the same policy without options.
- **M4 — CTDE critic:** expose global team state to the training critic, but keep actor inputs local at execution.
- **M5 — specialist replacement experiment:** only after stable interfaces and benchmarks exist, test separately trained aim/fire or movement experts, a learned gate, or hybrid neural + deterministic geometry.
- **M6 — opponent league and memory experiments:** evaluate each change against a frozen pool and matched seeds before accepting it.

A separate model should be retained only if its independent retraining/replacement advantage is real and its end-to-end team performance is at least as good. Keep a simple rollback path for every module.

## 4. Memory: three different problems

Memory is not a single feature. Separate the following layers:

### Working/temporal memory
Information from recent decisions that is not fully present in the current observation: target continuity, recent damage, whether an enemy disappeared behind an obstacle, an unfinished maneuver, a cooldown estimate, or an ally's recent intent.
- Cheapest prototype: explicit per-ship state with timestamps/expiry, e.g. last target, last seen time/position, current intent and intent expiry. This is inspectable and easy to test.
- Learned alternative: a GRU/LSTM recurrent policy. This requires sequence-aware rollout storage, recurrent-state reset at episode boundaries, sequence minibatches, and evaluation with partially hidden/incomplete observations. It is a meaningful architectural change—not just adding a tensor to the observation.
- Prefer tests that remove or hide selected observations; if memory improves performance there but not in the fully observed baseline, it is solving a demonstrated problem.

### Episodic memory
A compact log of specific previous encounters: composition, range/formation, opponent style, outcome, and useful tactical events. Retrieve a few similar cases or use them to select curricula/opponents. Do not continuously update policy weights during every battle just because a new episode happened; store experience and train/update in controlled batches.

### Long-term/semantic and procedural memory
- Semantic: reusable facts/patterns such as “this formation tends to overextend against kiting enemies,” if learned and validated.
- Procedural: reusable skills/options and frozen policy checkpoints.
- Experience replay is a training dataset, not the same thing as the agent's operational memory. Ordinary PPO is on-policy, so a generic replay buffer cannot just be inserted without algorithmic changes.

**Recommended first memory step:** explicit short-lived per-ship tactical state; second, recurrent GRU/LSTM only if evaluation proves useful; third, episodic opponent/encounter database and checkpoint pool. External memory need not be a huge neural memory module: a small, well-defined data structure may be more reliable and far easier to debug.

## 5. Brain analogy: useful, but not a literal blueprint

The brain is not one central executive surrounded by totally independent departments. It has specialized areas and pathways, but cognition is implemented by recurrent, distributed, interacting networks. “Executive control” is not simply one boss region that issues all commands.

A rough functional analogy—not a one-to-one map:
- **Prefrontal/frontoparietal networks:** maintain goals, working context, rule selection and flexible planning.
- **Basal ganglia / cortico-striatal loops:** help select/gate actions and learn habits and action values.
- **Hippocampal system:** binds events and context into episodic/spatial memory and supports retrieval.
- **Cerebellum:** predictive control, timing and fast motor adaptation.
- **Amygdala and related circuits:** learn biological relevance/threat and modulate responses; not simply a “fear center.”
- **Neuromodulatory systems:** influence learning, salience, exploration and plasticity via distributed signals.
- **Global-workspace theories:** propose that specialist processes operate in parallel and some information becomes globally available through a shared workspace. This is a theory of conscious access, not proof that any architecture with a broadcast bus is conscious.

The analogy is better expressed as interacting functions and feedback loops than as six replaceable neural nets. Functional specialization in brains coexists with shared representations, redundancy, and widespread connectivity.

## 6. “Consciousness”, internal states and synthetic feelings

For this project, distinguish:
- **Continuous autonomous control:** the simulation makes decisions each tick or on an event schedule without a human prompt.
- **Persistent goals/planning:** an internal state holds priorities and updates them using observations and results.
- **Self-monitoring/metacognition:** the agent estimates uncertainty, detects failure, and changes strategy or asks for more evidence.
- **Memory:** recent state, episodes, and reusable skills persist across decisions/battles.
- **Synthetic affect/motivation:** numeric internal variables such as threat, confidence, urgency, frustration-like failure counters, or resource pressure influence policy and persist for a defined duration.
- **Phenomenal consciousness/subjective experience:** there is “something it is like” to be the system. This remains scientifically unresolved for artificial systems; behaviour, emotion words, memory and self-reports alone do not establish it.

A functional “inner loop” with memory, priorities, simulated affect and short-/long-horizon planning is technically achievable as an engineering agent. It should be described as an *agent with internal state and control loops*, not as proof of consciousness or felt suffering. For GravitrueX, an abstract internal state can be useful if it changes measurable actions (e.g. raise retreat priority after repeated catastrophic damage) and can be evaluated. It does not need to be built into a sentient-style monolith.

## 7. Research references

- PPO, baseline RL: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Options / temporally extended actions: Sutton, Precup & Singh (1999), https://doi.org/10.1016/S0004-3702(99)00052-1
- Sparsely gated Mixture of Experts: Shazeer et al. (2017), https://arxiv.org/abs/1701.06538
- OpenAI Five architecture, LSTM, semantically meaningful action heads and team reward: https://openai.com/index/openai-five/
- Large-scale OpenAI Five training: https://openai.com/index/dota-2-with-large-scale-deep-reinforcement-learning/
- Recurrent policy for partial observations (DRQN): https://arxiv.org/abs/1507.06527
- Neural Episodic Control: https://proceedings.mlr.press/v70/pritzel17a.html
- Differentiable Neural Computer: https://deepmind.google/blog/differentiable-neural-computers/
- Hierarchical cortico-basal-ganglia behaviour and memory review: https://pmc.ncbi.nlm.nih.gov/articles/PMC6972576/
- Survey of LLM-agent memory mechanisms (2025): https://doi.org/10.1145/3748302
- “Identifying indicators of consciousness in AI systems” (Butlin et al.; 2025/2026): https://doi.org/10.1016/j.tics.2025.10.011
- Goldenberg & Gross, “Large language models do not have emotions” (Nature Human Behaviour, 24 August 2026; distinguishes emotion-like functions from subjective feeling): https://doi.org/10.1038/s41562-026-02558-6
- Anthropic, “A global workspace in language models” (6 July 2026): https://www.anthropic.com/research/global-workspace
- OpenAI, GPT-6 Astra (OpenAI model): https://openai.com/index/gpt-6-astra/
- Google DeepMind, Project Astra (a separate Google research-assistant project with a similar name): https://deepmind.google/models/project-astra/
- Nature commentary “Three reasons why AI is not conscious and perhaps never will be” (19 October 2026): https://www.nature.com/articles/d41586-026-03313-7

## 8. Decision rule

Do not pursue architecture because it sounds more human-like. Pursue it if a controlled evaluation shows an improvement in tactical behaviour, robustness, learning speed, memory efficiency, or debuggability. The target is a better game agent; “brain-like” is a source of hypotheses, not a performance guarantee.


## 9. New research direction: a minimal persistent cognitive agent (“proto-personality”)

**Motivation:** The goal is to test whether a small, modular system can develop a coherent, adaptive behaviour pattern over time—not to assume that a particular neural network size or diagram can create consciousness. The most useful first target is a persistent agent with memory, goals, self-monitoring, learning and a stable behavioural identity.

### Hard epistemic boundary: there is no validated minimum architecture for consciousness

As of October 2026, consciousness research has no agreed theory that establishes a small set of computational components as both necessary and sufficient for subjective experience. Global Neuronal Workspace Theory, recurrent-processing theories, higher-order approaches, Integrated Information Theory, predictive processing and memory-based approaches disagree about key requirements. A 2026 integrative review explicitly notes that the minimum complexity threshold and the relevant organizational properties remain unsettled. Butlin et al. (published online November 2025; journal issue June 2026) propose evaluating AI through theory-derived indicators, while stressing that the indicators are not individually necessary or jointly proven sufficient.

Therefore:
- Do not claim that adding memory, an inner loop, emotions, a self-model or a workspace creates consciousness.
- Distinguish **functional agency** (the system observes, chooses, acts, remembers and adapts), **proto-personality** (stable preferences and behavioural tendencies over time), and **phenomenal consciousness** (subjective experience).
- The first two are tractable engineering goals with observable tests. The third remains an open scientific/philosophical question.
- “Minimum necessary” should be treated as a research question and ablation programme, not as a known recipe.

### Smallest practical architecture worth trying

Avoid beginning with a huge Transformer, many layers, or five independently trained networks. Start with a modest recurrent core plus explicit state and memory. The system can use ordinary program logic for bookkeeping and a small trainable model for decisions; not every function needs to be another neural net.

**A. Perception / world state.** Convert current observations into a compact structured state: what is happening, what is uncertain, what changed, and which entities/events matter. For a simple simulation this can be numeric features plus a short event list; in a richer environment it can be a learned encoder.

**B. Recurrent working core.** A small GRU (or LSTM) receives the current state, the previous internal state, the active goal, relevant retrieved memories, and feedback on the last action. It returns an updated hidden state. This is the candidate for integrating information across time; it is not itself evidence of consciousness. A small starting experiment might use one GRU layer with 128–256 hidden units, but this is a tunable engineering baseline—not a theoretically sufficient size.

**C. Shared workspace / attention selection.** Keep a compact “currently relevant” record that selected modules can read and write: current situation summary, active goal, main uncertainty, current plan, important memory retrieved, and last action result. An attention/selection mechanism decides what deserves this shared workspace, and the selected information is available to the planner, action policy and self-monitor. It may initially be a deterministic top-K selector; learnable attention can come later.

**D. Persistent goal stack and value signals.** Store one long-term objective, a few subgoals, priorities, constraints and termination conditions. Numeric drive/affect-like variables may represent urgency, threat, curiosity, confidence, fatigue/resource pressure or recent failure. These values should have explicit update rules and observable effects on decisions. They are functional control signals—not claims of felt emotion. Avoid unconstrained goal rewriting; require reasons/evidence for switching goals and use hysteresis to prevent oscillation.

**E. Self-model and world model.** Track what the agent can do, its resource/health state, its current skills and uncertainty, what it believes about the environment, and what outcome it predicts for a candidate action. Initially, predictions can be simple learned estimates or explicit simulation; later, test a learned dynamics/world model. Self-reports should be checked against actual logs so the agent can be wrong about itself and correct that belief using evidence.

**F. Action selection and hierarchical planning.** A low-level policy handles immediate actions. A slower planner selects a goal/option and may simulate a few candidate outcomes, then return control to the policy. The high-level process should run less often than the motor loop. Start with a handful of behaviours/options instead of unconstrained open-ended planning.

**G. Working and episodic memory.** Working memory holds the present task, intention, recent events and countdowns. Episodic memory stores compact event records: situation, chosen action, predicted result, actual result, lesson/uncertainty, and context. Retrieval returns a few relevant records, not the whole lifetime log. Persistent memory should be versioned, editable and logged so corrupted or contradictory memory can be diagnosed.

**H. Metacognitive monitor.** Compare predictions with outcomes; track confidence calibration, repeated failure, stuck loops, uncertainty and contradictions. When necessary, request another observation, retrieve a memory, change strategy or defer to a safe fallback. A monitor that merely generates convincing explanations without checking evidence does not count.

**I. Learning and consolidation.** Store experience continuously, but update policy/model weights in controlled training windows, validate against old tasks and keep checkpoints. The system can consolidate repeated events into a more general rule or skill, but a saved memory, a verbal reflection, and a genuine improvement in policy performance are three different things.

### Minimal data flow

At every decision cycle:

1. Observe the world and record relevant new events.
2. Update beliefs, uncertainty and the agent’s own state estimate.
3. Retrieve relevant memories and update working state.
4. Update priority/drive signals and decide whether to keep or revise the active goal.
5. If needed, run a higher-level planning step; otherwise continue the active plan.
6. Choose an action; record the prediction/rationale in a compact structured form.
7. Act, then compare the actual result with the predicted result.
8. Update immediate memory and evaluation signals; schedule controlled learning/consolidation separately.

This is a recurrent feedback architecture, not a requirement to generate a stream of inner-language narration on every frame. Natural-language self-dialogue could be added later as an interface, but it is expensive and can create the illusion of introspection without improving decisions.

### What would count as a useful “personality” in this project?

Define personality operationally as **persistent, measurable behavioural tendencies that are learned or deliberately configured, influence decisions across episodes, and remain coherent under changing circumstances**. Potential dimensions include risk tolerance, exploration preference, loyalty/escort priority, curiosity, patience, or preference for preserving resources. Keep a separation between:
- stable baseline preferences / values;
- transient affect-like state;
- temporary task goals;
- learned skills and facts;
- uncertainty and current beliefs.

A system should not be assigned a trait merely because it says it has one. Verify it by behaviour. For example, if it claims to be cautious, controlled tests should show that it accepts less risk under comparable conditions while still pursuing goals. Its traits may evolve through training, but changes should be measured and reversible.

### Lowest-effort prototype plan

**P0 — environment and logging.** Use a tiny, fast environment with partial observability, delayed consequences, a few meaningful resources, and repeated episodes. Start with a minimal grid-world or simple 2D arena, not the full GravitrueX combat pipeline. Save every observation, goal, memory retrieval, action, prediction, outcome and model version.

**P1 — baseline agent.** Train a small recurrent actor-critic or GRU policy; compare it against a feed-forward policy with the same parameter budget. The recurrent agent should only win if history actually helps. Keep a clean evaluation suite and fixed seeds.

**P2 — explicit memory and goals.** Add a goal stack and compact episodic store. Give the agent tasks where a remembered key/location, previous failure, or delayed objective is useful. Test retrieval and goal persistence independently of model size.

**P3 — self-monitor and simple internal drives.** Add uncertainty, repeated-failure and urgency signals that alter behaviour in predefined ways. Ablate each variable to see whether it improves results rather than simply making logs more expressive.

**P4 — adaptation and consolidation.** Allow a controlled learning update between episodes, then evaluate old tasks to detect forgetting. Keep an immutable original checkpoint and compare adapted vs. non-adapted agents.

**P5 — gradual complexity.** Only after P0–P4 demonstrate real gains, add a learned world model, more powerful planner, specialist modules, language interface or broader environment. Do not scale model size until a controlled ablation shows that capacity is the bottleneck.

### Acceptance tests: prove capability, not a consciousness claim

- **Delayed-goal test:** can the agent return to an earlier objective after an interruption?
- **Memory test:** does a relevant stored episode improve choices on a new but related situation?
- **Learning test:** after repeated failure and a learning update, does the failure rate decrease on held-out seeds?
- **Counterfactual prediction test:** does the agent’s predicted outcome correlate with what actually happens?
- **Self-model calibration:** when it estimates a high/low chance of success, do observed outcomes support that calibration?
- **Consistency test:** do stable traits influence behaviour across varied contexts without turning into rigid rules?
- **Transfer test:** does a learned rule help in a related environment with changed surface details?
- **Forgetting/regression test:** does new learning preserve prior competence?
- **Ablations:** remove recurrence, memory, goal persistence, self-monitor or drive signals one at a time. This tests which components actually contribute.
- **No-personality-theatre test:** compare logged actions and decisions, not the agent’s verbal claims about what it supposedly feels or remembers.

Success on these tests would establish useful functional capacities and a coherent agent. None would by itself establish subjective experience.

### Decision for GravitrueX versus a separate prototype

Do not directly splice a cognitive architecture into the current large PPO model while its performance and telemetry are still being stabilized. Keep the notebook as the shared design reference, finish the current training baseline first, and build a small sandbox prototype alongside it. Once the memory/goal loop is demonstrated, selected mechanisms—persistent target/intent memory, a tactical goal selector, recurrent policy, or meta-monitor—can be ported into GravitrueX one at a time.

### Research references for this direction

- Butlin et al., “Identifying indicators of consciousness in AI systems,” *Trends in Cognitive Sciences* 30(6), 2026; online 10 Nov 2025. https://doi.org/10.1016/j.tics.2025.10.011
- Butlin et al., “Consciousness indicators, mimicry, and internal variants,” *Trends in Cognitive Sciences*, July 2026. https://doi.org/10.1016/j.tics.2026.04.006
- Whyte et al., “On the minimal theory of consciousness implicit in active inference,” *Physics of Life Reviews* 56, 2026. https://doi.org/10.1016/j.plrev.2025.11.002
- “Consciousness in the Brain: An Integrative Review of Contemporary Theories,” *Brain Sciences* 16(7), 2026. https://doi.org/10.3390/brainsci16070745
- Soar cognitive architecture manual (working memory, production/semantic memory, preferences, decision cycle, learning): https://soar.eecs.umich.edu/soar_manual/02_TheSoarArchitecture/
- Cognitive architecture of anticipatory thinking, *AI Magazine* (perception, working/long-term memory, decisions, learning and action): https://doi.org/10.1002/aaai.12102
- Bayesian theories of consciousness review (predictive processing / active inference; a theoretical position, not established consensus): https://doi.org/10.1093/nc/niab038
