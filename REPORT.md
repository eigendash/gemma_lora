# Tool-Use Reasoning from a Tool Description: GRPO with LoRA on Countdown

## Summary

**Setup.** We train LoRA adapters (r = 16) on base language models with GRPO on the Countdown numbers game. A
rollout may reason inside `<think>`, run Python inside `<python>`, read the sandbox's real output inside `<output>`,
and must commit to an expression inside `<answer>`. Injected output tokens are masked out of the loss. The reward is
correctness from an exact verifier, plus small terms for working code and well-formed tags, minus a penalty on length
beyond 512 tokens. Every run uses the same 300-step schedule, and every final checkpoint is scored once on holdout3, a
1,000-problem set generated after all training and selection decisions were made.

| Model | Training | Prompt | holdout3 pass@1 |
|---|---|---|---|
| Qwen3-4B-Base | none | demo | 9.4% |
| Qwen3-4B-Base | none | zero-shot (`r1zero_tool`) | 6.7% |
| Qwen3-4B-Base | recipe v1, three lineages | demo | 49.8–52.1% |
| Qwen3-4B-Base | recipe v2 | demo | 68.1%; 68.2% after 45 more steps |
| Qwen3-4B-Base | recipe v2 | zero-shot | **69.9%**; 70.5% after 45 more steps |
| Qwen3-8B-Base | recipe v1 | demo | 55.7% |
| Gemma-4-E4B | recipe v2 | zero-shot | 60.5%; 68.1% after 100 more steps |
| Gemma-4-E2B | recipe v2 | zero-shot | 4.4% |
| Gemma-4-E2B | recipe v2 | demo | 16.3% (stopped at step 82) |

**Findings.**
1. *The recipe matters more than model size.* Recipe v2 drops the KL penalty, hinted resampling and training on
   groups with no correct rollout, and adds gradient clipping. It improves on the v1 replay with the same seed by 18
   points (95% interval 15–22, p < 10⁻²⁷), more than moving from Qwen3-4B to Qwen3-8B under v1 (Section 2.1).
2. *A description of the tool is enough.* A zero-shot prompt that only says what the tags do, with no worked
   example, trains Qwen3-4B to 69.9%, level with the arm trained on a demonstration (68.1%, p = 0.23). Giving both
   arms 45 more steps leaves them level (70.5% and 68.2%, p = 0.13). The result survives a rewording of the prompt
   (69.6%), though a second, terser rewording costs 3 points (Section 2.2).
3. *RL extends what the model can solve, not just how often it samples it.* At temperature 0.9, the zero-shot policy
   solves 67% of test problems in one sample; the base model, with the same prompt, solves 20% in eight (Section 3).
4. *The zero-shot policies use the tool mainly to confirm.* With code execution disabled, the v2 zero-shot policy
   loses 6 points and Gemma-4-E4B 1 point, against 37 for the demo-trained v2 policy and 34 for the v1 policy. They
   do read the output: falsified output costs them 11 and 13 points. The demo-trained policy is the one that searches
   with code and recovers after a failed check (Section 3).
5. *The result transfers to Gemma-4-E4B, but not to 2B.* Gemma-4-E4B reaches 60.5%, and 68.1% after 100 more steps
   (p ≈ 10⁻⁹), by which point it runs code on all but one problem. Gemma-4-E2B starts learning zero-shot, but its
   sampling collapses to one answer per problem within 50 steps and it ends at 4.4%. With the demonstration it
   reaches 16.3% before its responses outgrow the token budget (Section 2.3).

**Learning dynamics, backtracking, compute and exploits.** Reward, validation accuracy and length over training are
in Figure 2 (Section 2.4); a verbatim backtracking trace is in Section 3. The one reward exploit we saw, answering
straight after a failed check to collect the shaping reward, and the safeguards that closed it are in Section 4.
Everything ran on single spot GPUs with colocated vLLM generation, at a total of 144 GPU-hours (Section 6). Section 5
gives the comparison on the original 1,000-problem test set that the task requests, for every model.

**Main caveat.** Each recipe v2 arm on Qwen3-4B has one completed seed, and it is the same seed for both. With seed 1,
both arms were stopped in phase b by the guard on truncated responses. Recipe v2 as configured is therefore
sensitive to the seed, and the 2-point run-to-run variation measured under v1 probably understates its variance. We
claim parity between the two prompts, not that either is better (Section 7).

## Abstract

We study whether reinforcement learning can teach a small base language model to solve arithmetic search problems
by reasoning, writing code, reading the code's real output, and revising its answer. We train LoRA adapters with
Group Relative Policy Optimization (GRPO) on the Countdown numbers game. The model may interleave free-form
reasoning, Python execution in a sandbox, and a final answer, and the environment's output is injected into its
context during generation but excluded from the loss. The main result concerns the role of the prompt. We had
concluded early on that zero-shot training was impossible, because Qwen3-4B-Base scored 0% on the test set when
given only the tag skeleton. The cause turned out to be that this prompt showed the output format but never said that
code would be executed. With a zero-shot prompt that simply describes the tool, the base model scores 6.7% on a fresh
1,000-problem holdout set, and 69.9% after 300 GRPO steps. A policy trained with a worked demonstration under an
otherwise identical configuration reaches 68.1% (paired McNemar p = 0.23), and the two stay level when both are
trained for 45 more steps. The two policies use the tool differently. The demo-trained policy searches with code and
loses more than half its accuracy when execution is disabled; the zero-shot policy solves most problems in its
reasoning, runs code to confirm the answer, and loses 6 points without it. Its single-sample accuracy at temperature
0.9 (67%) is well above the base model's pass@8 (20%). The recipe transfers to Gemma-4-E4B, which reaches 60.5% after
300 steps and 68.1% after 100 more; by then it runs code on almost every problem but gains only a point from it. At 2B
parameters the zero-shot recipe starts but stalls. A second seed of the zero-shot arm, like a second seed of the demo
arm, was stopped by a guard on response length, so the recipe is less stable across seeds than our single completed
seed per arm suggests. We also describe the one reward exploit we observed, the safeguards built against it, and the
compute measures that made 144 GPU-hours of experiments feasible on single spot GPUs.

## 1. Method

### 1.1 Task, data and evaluation protocol

Each problem gives three to six numbers and a target. The model must reach the target with `+`, `-`, `*` and `/`,
using each number at most once. A generator built around an exact solver produces 10,000 training problems and a
1,000-problem test set, all guaranteed solvable. The problems come in three tiers, mixed in proportions 40/30/30:
- **Easy:** three or four numbers with a two-digit target.
- **Medium:** five numbers with a three-digit target.
- **Hard:** six numbers with a three-digit target.

We hold out 300 training problems for validation during training. Over the course of the project we also generated
three further 1,000-problem holdout sets, each after a round of model selection and each disjoint from all data
generated before it. All model comparisons in this report use the last of these, holdout3. It was created after
every training and checkpoint decision had been made. Each final checkpoint was evaluated on it exactly once, with
greedy decoding, a budget of 768 new tokens and at most two code executions per problem. We report 95% Wilson
intervals for single models. Models are compared with exact McNemar tests on the problems where they disagree, and
differences between them carry 95% paired bootstrap intervals (10,000 resamples of the 1,000 problems).

### 1.2 Interaction protocol and sandbox

A rollout follows the protocol `<think> … </think>`, optional `<python> … </python>` blocks, and a final
`<answer> … </answer>`. When the model closes a `<python>` block, generation pauses. The code is executed, and its
standard output is inserted inside `<output> … </output>` before generation resumes. Tokens in the inserted output
carry label −100, so the policy is never trained to predict the environment. The rollout engine feeds the model
exactly the token sequence used for training, which rules out tokenisation drift at the boundaries between generated
and inserted text. If the model writes an `<output>` block itself instead of running code, the rollout ends
immediately.

Code is first parsed and checked against an AST whitelist. It allows arithmetic, comparisons, assignments and loops,
and calls only to `print`, `range` and `len`; imports, attribute access and file or network I/O are rejected. Code
that passes runs with the same three builtins in a persistent, stdlib-only worker process (`python -I -S`). The worker
is killed and respawned if execution exceeds 50 ms, and output beyond 200 characters is discarded. We chose process
isolation over the thread-based design suggested in the task description, because it protects the training process
from runaway loops and memory exhaustion. The adversarial tests and the 10,000-execution stress test in the
repository exercise these cases.

### 1.3 Reward

The reward has four terms:

R = R_correct + R_exec + R_format − λ · max(0, L − 512).

- **R_correct (1.0)** is assigned by a verifier that is independent of the model. It parses the answer into an AST,
  evaluates it with exact rational arithmetic, and checks that the numbers used are a sub-multiset of those given.
- **R_exec (0.1)** requires at least one code block that ran without error, printed something, and contains an
  arithmetic expression combining at least two of the given numbers. Code such as `print(target)` therefore earns
  nothing.
- **R_format (0.1)** rewards correctly ordered and closed tags around a well-formed expression.
- **The length penalty** (λ = 0.001 per token) applies only beyond 512 generated tokens.

Section 4 describes two further conditions under which the shaping terms R_exec and R_format are withheld.

### 1.4 Policy optimisation

We use GRPO with groups of G = 8 rollouts for each of 64 prompts per step. Advantages are normalised within each
group, and the loss is a token-level clipped surrogate with an asymmetric range (0.2 below, 0.28 above), applied in
two updates per batch. Rollouts are generated with vLLM running on the same GPU as the trainer. The small mismatch
between the sampler's and the trainer's log-probabilities is corrected with truncated importance weights capped at 2.

The adapter is LoRA with rank 16 and α = 32, trained at a constant learning rate of 2 × 10⁻⁵. A curriculum chooses
each problem's tier in proportion to the fraction of that tier's recent groups whose outcomes were mixed (between 1
and 7 of the 8 rollouts correct). Only such groups have non-zero advantages, so the curriculum concentrates sampling
where gradient signal exists. Groups with no variance are refilled with fresh prompts, up to two extra rounds per
step.

We chose this signal over the policy's entropy, which is the other common basis for a curriculum, because it measures
directly what GRPO needs: a group contributes a gradient if and only if its outcomes differ. Entropy is only a proxy
for that. A tier can have high entropy and no correct rollouts at all, which yields no gradient under recipe v2, or
low entropy with a reliable mix of outcomes. The proxy does carry information that our curriculum ignores, however.
The Gemma-4-E2B run in Section 2.3 lost its entropy, and with it nearly all its signal, within 50 steps. The
mixed-outcome fraction registered the loss only after it had happened, whereas a fall in entropy would have flagged
it earlier.

Every run follows the same four-phase schedule of 40, 90, 90 and 80 steps (300 in total). Each phase starts from the
final checkpoint of the previous one, so no checkpoint is chosen by hand.

### 1.5 Prompts

We use three prompts. All of them state the rules and the output format; they differ in how much they show the model
about using the tool.

| Prompt | Used for | What the model sees |
|---|---|---|
| `fewshot_direct_retry` | demo arms, phase a | One worked example: a single candidate expression checked with `print(res == target, res)`, output `True`, answer. Plus the instruction to try a different expression if the check prints `False`. |
| `fewshot_multi_retry` | demo arms, phases b–d | The same, but the example checks four candidates in one loop and answers with the one that prints `True`. The instruction asks for several candidates per block and new ones if none succeeds. |
| `r1zero_tool` | zero-shot arms, all phases | The R1-Zero conversation frame, a one-sentence description of the tool ("code inside `<python>` tags is executed and its printed output is returned inside `<output>` tags"), the allowed operations, the limit of two executions, and an example of the answer syntax. There is no worked problem, no checking strategy and no instruction to retry. |

A fourth prompt, `zero`, appears only in the baseline evaluation. It gives the tag skeleton but no description of
what the tags do. We refer to the two few-shot prompts together as the demo prompt.

### 1.6 Training recipes

We compare two configurations of the optimiser.

| | Recipe v1 | Recipe v2 |
|---|---|---|
| KL penalty to the base model | adaptive (it stays at its floor of 0.001) | none |
| All-fail groups in phases c and d | resampled once with a partial-solution hint | not resampled |
| Groups with no correct rollout | trained on (shaping terms only) | discarded |
| Gradient clipping | none | global norm 1.0 |
| Order of the two updates per batch | fixed | shuffled |

Everything else is identical in the two recipes: the reward, the learning rate, the batch shape, the curriculum,
the schedule and the token budget.

Recipe v1 grew out of a sequence of hand-tuned phases on Qwen3-4B. It was later replayed from scratch with other seeds
and models to separate the recipe from the selection effects of its own development. Recipe v2 removes the components
that our ablations showed to be inert or harmful. Hinted resampling had no measurable effect: phase d without hints
scored 518 against 513 with them on an earlier holdout (p = 0.77). The KL penalty was effectively inactive. Training
on groups that contain no correct rollout gives the shaping terms a large share of the gradient when accuracy is low,
which is the mechanism behind the exploit in Section 4.

## 2. Results

![Figure 1](docs/figures/fig1_holdout3.png)

*Figure 1. Accuracy on holdout3 (1,000 problems generated after all model decisions) for every final checkpoint, with
95% Wilson intervals. Each checkpoint was evaluated once, with greedy decoding. The Gemma-4-E2B demo row is the
checkpoint at which the length guard stopped that run, 82 steps into the schedule (Section 2.3); the Qwen3-4B
seed-1 zero-shot row is likewise the last checkpoint of a run the guard stopped, after 53 steps (Section 2.2).*

### 2.1 Recipe v2 improves substantially on recipe v1

Three independent v1 lineages of Qwen3-4B reach between 49.8% and 52.1% on holdout3 (Fig. 1). Two of them are
from-scratch replays with seeds 1 and 2, and the third is the original hand-tuned lineage. Their spread of about
2 points is our best estimate of run-to-run variation. With the same seed and schedule, recipe v2 reaches 68.1%,
against 49.8% for the matching v1 replay, a difference of 18.3 points (95% interval 15.1–21.5). The two models
disagree on 293 problems, and v2 is right on 238 of them (p < 10⁻²⁷). Qwen3-8B trained with v1 reaches 55.7%, 12.4
points below v2 on the 4B model (p < 10⁻¹⁴), so the change of recipe is worth more than doubling the model.

The training logs suggest why. Under v1, a group in which all eight rollouts are wrong still produces a gradient
through the format and execution terms. That gradient favours short, well-formed failures, and so acts as an implicit
length regulariser. Without it, v2's rollouts first grew longer: one of the two v2 seeds was stopped in phase b by our
pre-registered guard on truncation (the zero-shot arm later did the same with that seed; Section 2.2). The other seed
recovered as accuracy rose, and its share of capped rollouts fell from 32% to 6% by the end of training. Continuing
that run for another 45 steps left holdout3 accuracy unchanged (68.2%), so for this run the four-phase schedule had
already converged. That is not true of every model: the same check on Gemma-4-E4B found a large further gain (Section
2.3).

### 2.2 A tool description is sufficient

All the runs above used the demo prompt. To test whether the demonstration is necessary, we trained a matched
zero-shot arm. It is identical to the v2 demo run in seed, schedule and every hyperparameter, but uses `r1zero_tool`
in all four phases. Before any training result, we registered the following reading: if the zero-shot arm scored
661 or more on holdout3 (within two points of the demo run's 681), we would conclude that the demonstration is not
needed.

The choice of prompt matters a great deal for whether training can start. Under v2, only groups containing at least
one correct rollout produce a gradient. With the `zero` prompt, base Qwen3-4B solves at least one of eight samples on
only 3 of 256 training problems, and 38% of its answers simply restate the target. With `r1zero_tool`, 58 of 256
problems have at least one correct sample. That is enough signal for training to begin.

On holdout3 the base model scores 6.7% with `r1zero_tool` and 9.4% with the demo prompt, so neither prompt on its own
accounts for much of the final accuracy. The zero-shot arm starts 11 points behind the demo arm on validation (4.3% vs
15.0%), draws level within roughly 20 steps, and stays at or above it for the rest of training (Fig. 2a). On
holdout3 it scores 69.9%, against 68.1% for the demo arm, which meets the registered criterion. The difference of
18 problems (109 vs 91 discordant; 95% interval −0.9 to +4.5 points) is not significant, so we read the result as
parity rather than superiority. Almost all of the difference comes from the hard tier (181 vs 153).

Three follow-ups, registered after this result, test how far it holds.

*Training length.* The demo arm had been continued for 45 steps, so we gave the zero-shot arm the same 45 steps. It
moved from 69.9% to 70.5% (p = 0.69), a plateau under the registered 2-point criterion, as the demo arm's
continuation had been. At an equal 345 steps the two arms stand at 70.5% and 68.2% (p = 0.13; interval −0.6 to
+5.1). Parity therefore does not depend on where the schedule happened to stop.

*Wording.* The policy is closely tied to the format of its prompt: evaluated with the demo prompt instead of its own,
it falls to 30.9%. We also wrote two rewordings of `r1zero_tool` that carry the same information in different
sentences, and evaluated the final checkpoint with each. The first stays close to the original's structure and scores
69.6% (p = 0.86 against the original). The second is terser and gives the tags as a list; it scores 66.6%, a small but
real loss (−3.3 points, interval −5.6 to −0.9, p = 0.008), almost all of it on the medium and hard tiers. The policy
has learned the task rather than the exact text, but it has not become indifferent to how the task is put.

*A second seed.* We trained the zero-shot arm with seed 1, on which the demo arm had been stopped in phase b by the
length guard. Phase a went well: validation rose from 5.3% to 41.0% within 30 steps, slightly ahead of seed 2 (36.0%).
But the share of rollouts reaching the 768-token budget rose with accuracy, from 14% to about 40% by the end of phase
a, and the guard stopped the run 13 steps into phase b. Seed 2 had run just under the same line, with single steps up
to 43% but a five-step mean that stayed below 40%, and its share then fell as accuracy rose. The last seed-1
checkpoint, after 53 steps, scores 49.7% on holdout3, a third of its responses truncated. For reference, the seed-2
checkpoints at the end of phase b (130 steps) score 58.2% (zero-shot) and 57.1% (demo; p = 0.48). Both prompts thus
trip the guard on seed 1 and complete on seed 2. Whether a v2 run finishes depends on the seed more than on the
prompt, and the guard's threshold lies within the range that the seed produces.

### 2.3 Transfer to Gemma

We ran the same zero-shot configuration on Gemma-4-E4B, using its text decoder extracted from the multimodal
checkpoint. The base model scores 1.9% on holdout3 with `r1zero_tool`. After the 300-step schedule it reaches 60.5%,
compared with 30.9% for Gemma under recipe v1 with the demo prompt. We give no p-value for that comparison, because
the recipe and the prompt both differ. At that point its tool use is very different from Qwen's. Gemma runs code on
only 35% of problems, and no first execution ever prints the target. Instead, it enumerates combinations in its
reasoning and turns to code only once it is stuck, which mostly happens on problems it goes on to fail.

Validation accuracy was still rising at the end of the schedule, so we registered a 100-step continuation, with a
gain of at least 2 points on holdout3 as the criterion for "still improving". The continuation reached 68.1%, a gain
of 7.6 points (interval 5.2–10.0; 117 problems gained and 41 lost, p ≈ 10⁻⁹), spread over all three tiers. Its
behaviour changed more than its accuracy did. The continued policy runs code on 999 of the 1,000 holdout3 problems,
against 345 before, and averages 1.1 executions per problem instead of 0.5. What it learned to do with the tool is
narrow, however. Its first check prints the target on 677 of those 999 problems, so the code almost always confirms
an answer already reached in its reasoning. When the first check misses, it solves only 5 of the 322 problems
concerned. With execution disabled altogether it loses about one point on the test set (Section 3). Running a check
became a habit; relying on its result did not. Like the Qwen policy, Gemma is tied to its training prompt: with the
demo prompt the 300-step checkpoint scores 42.7% instead of 60.5%.

At about 2B parameters the starting signal nearly disappears. Gemma-4-E2B solves at least one of eight samples on 1
of 256 problems, and gemma-2-2b, the model suggested in the task description, on none. We nevertheless ran two
matched 50-step runs on Gemma-4-E2B, one with each prompt, each gated on a registered criterion of at least 3 points
of validation gain. The zero-shot run passed its gate. It found almost no signal for the first 20 steps (0–2 of 64
groups per step), after which signal and accuracy rose together, reaching 6.0% validation accuracy at step 50
(from 0.3%). Over the same 50 steps Qwen3-4B went from 4.3% to 49.3%.

The rise came at the cost of diversity. The policy's sampling entropy fell from 0.77 to 0.10 over phase a, and by its
end 77% of prompts produced eight identical rollouts. A group of identical rollouts has no advantage to learn from,
so we continued the run through the remaining three phases to see whether it would recover. It did not. Over 260
further steps validation stayed at exactly 6.0%, with identical per-tier counts, and most steps contained no group
with a gradient at all. The final checkpoint scores 4.4% on holdout3.

The demo-prompted run behaved quite differently. From step 0, 77% of its answers repeated an expression that its
own check had just shown to be wrong, which stopped the run at step 5 on the guard against reward exploitation
(Section 4). Under recipe v2 such answers earn no gradient, so the guard protects nothing in this recipe, and we
reran the arm with it disabled. That run passed the same gate comfortably, rising from 1.0% to 11.4% validation
accuracy in 50 steps with 6–43 signal groups per step throughout, and contradicted answers fell steadily. It then
continued into phase b, where its responses grew longer as accuracy rose (16.7% at step 30). After 32 steps, 40% of
its rollouts were hitting the 768-token budget, and the length guard stopped it. This is the same failure that
stopped the seed-1 Qwen3-4B v2 runs. Evaluated at that point, 82 steps in, it scores 16.3% on holdout3, against 4.4%
for the zero-shot arm after 310 steps (121 problems gained and 2 lost). Almost all of the gain is on the easy tier
(138 against 36 of 400), and over half of its holdout3 responses are truncated.

At this scale, then, the demonstration is what keeps learning going. It brings the model to an accuracy at which
groups still contain both successes and failures before its sampling narrows. Without it, the policy narrows while
nearly every rollout still fails. These runs, together with an early comparison in which gemma-2-2b solved about 2%
of problems with the demo prompt against about 15% for Qwen3-4B, are why the study uses Qwen3-4B as its main model.

![Figure 2](docs/figures/fig2_dynamics.png)

*Figure 2. Training dynamics of Qwen3-4B under recipe v2 (seed 2) with the demo prompt and the zero-shot prompt.
Vertical lines mark the phase boundaries at steps 40, 130 and 220. (a) Greedy accuracy on the 300 validation
problems, measured every 5 steps. (b) Mean training reward. (c) Mean number of generated tokens per rollout, excluding
inserted output. Faint lines are per-step values and bold lines 7-step moving averages. Breaks are steps missing from
the logs (Section 7).*

### 2.4 Learning dynamics

In both arms, reward and validation accuracy rise together over the 300 steps and are still rising slowly at the
end (Fig. 2a, b). The training reward understates the progress, because the curriculum shifts sampling toward harder
problems as the easier tiers saturate.

The two arms differ most in length (Fig. 2c). The demo arm stays at about 180 tokens during phase a, then jumps to
about 460 at step 40. That is when its prompt switches to checking several candidates per block. The zero-shot arm
receives no such instruction, yet it lengthens on its own to about 470 tokens within its first 30 steps, and it
shortens to about 350 tokens as its checks become more targeted. Here the length penalty did real work. In phase a,
26% of zero-shot rollouts ran into the 768-token budget. The penalty beyond 512 tokens brought that down to 1% by
phase d without any explicit length target. The demo arm's share of capped rollouts peaked at 33% in phase b and
ended at 10%.

## 3. Self-correction after negative feedback

![Figure 3](docs/figures/fig3_self_correction.png)

*Figure 3. What follows the first code execution on holdout3, for the two Qwen3-4B v2 policies. A first check
"finds the target" if its output prints `True` or a number equal to the target. The remaining problems are split by
whether the model ran a second check and whether it then solved the problem.*

The two v2 policies use execution feedback differently (Fig. 3). When the first check misses, both retry at almost the
same rate: 74% for the zero-shot policy and 75% for the demo policy. Of the problems it retries, the demo-trained
policy goes on to solve far more (59% vs 22%; counted over all first-check misses, retried or not, the rates are 50%
and 18%). The zero-shot policy's overall advantage comes from missing much less often in the first
place: 230 first-check misses against 619. It does the arithmetic in its reasoning and runs code mainly to confirm a
result it has already derived.

Its retries are nonetheless genuine backtracking. In most recoveries, the printed value shows that the expression it
wrote does not match its own reasoning, through a misread sum or a flipped sign. The model then abandons that
decomposition and builds a new one. A typical hard example from holdout3 (verbatim, whitespace condensed):

```
numbers 6 7 2 25 1 8, target 168
<think> 6 * 28 = 168. 28 can be made from 25 + 7 - 2 = 30 - 2 = 28 ... </think>
<python> print(6 * (25 + 7 - 2)) </python>          <output> 180 </output>
<think> That didn't work, let's try another combination: 8 * 21 = 168. 21 can be made from 7 * 3,
        and 3 can be 6 - 2 - 1 ... </think>
<python> print(8 * (7 * (6 - 2 - 1))) </python>     <output> 168 </output>
<answer> 8 * (7 * (6 - 2 - 1)) </answer>
```

**How much the policies depend on the tool.** Retrying after a miss shows that a policy reads the output, not that
it needs it. To measure the dependence we intervened on the tool at evaluation time, on the test set with the HF
backend: we limited the number of code executions, replaced the inserted output with a blank, or shuffled the digits
of the output.

*Table 1. Test-set accuracy (%) under interventions on the tool. Greedy decoding, 768 tokens; "normal" allows two
executions with real output.*

| Policy | Normal | No code | One execution | Output blanked | Digits shuffled |
|---|---|---|---|---|---|
| Qwen3-4B, v1 hand-tuned (demo) | 51.8 | 17.8 | 47.3 | 12.8 | 47.4 |
| Qwen3-4B, v2 demo | 69.2 | 32.3 | 64.9 | 58.5 | 65.2 |
| Qwen3-4B, v2 zero-shot | 70.7 | 64.5 | 68.4 | 65.5 | 59.8 |
| Gemma-4-E4B, v2 zero-shot, +100 steps | 70.2 | 69.0 | 70.2 | 70.1 | 56.8 |

The four policies fall into two groups. The demo-trained policies depend on the tool. Without code, the v1 policy
keeps a third of its accuracy and the v2 demo policy less than half. The v1 policy does worse still with a blank
output than with no code at all, because it waits for a `True` that never comes; shuffled digits leave the flag
intact and cost it little. The v2 demo policy copes better with a blank (58.5%), so by the end of training it has
some ability to reason without the result.

The zero-shot policies need the tool much less. Without any execution, the Qwen policy loses 6 points and the Gemma
policy one. Both nevertheless read what the tool returns and believe it: shuffled digits cost them 11 and 13 points,
more than a blank output does. A wrong number misleads them, whereas a missing one sends them back to their own
reasoning. This fits the miss counts in Fig. 3. The zero-shot policy derives the answer before it runs code, and the
code mostly confirms it. The behaviour that emerged without a demonstration is verification, not search with code.
The demonstration teaches search, because it shows several candidates checked in one block; that is also what made
the demo policy good at recovering from misses.

**Sampling.** A common objection to RL on verifiable rewards is that it only concentrates probability on solutions the
base model already samples. We sampled eight completions per test problem at temperature 0.9 and computed unbiased
pass@k.

*Table 2. Test-set pass@k (%), temperature 0.9, eight samples per problem (four for the v1 policy and the
demo-prompted base model).*

| Model | Prompt | pass@1 | pass@2 | pass@4 | pass@8 |
|---|---|---|---|---|---|
| Qwen3-4B base | `r1zero_tool` | 3.6 | 6.8 | 12.1 | 20.0 |
| Qwen3-4B base | demo | 9.9 | 17.2 | 27.6 | |
| Qwen3-4B, v1 hand-tuned | demo | 49.5 | 57.5 | 65.3 | |
| Qwen3-4B, v2 zero-shot | `r1zero_tool` | 67.2 | 75.8 | 82.4 | 87.6 |
| Qwen3-4B, v2 demo | demo | 68.2 | 78.0 | 85.3 | 90.1 |

The zero-shot policy's single sample solves more than three times as many problems as eight samples of the base
model with the same prompt. At the sample sizes we can afford, training does more than sharpen the base
distribution; we cannot exclude that the base model would close the gap at much larger k. The trained policies keep
useful diversity: pass@8 is about 20 points above pass@1 for both. The demo policy gains more from extra samples
(90.1% against 87.6% at k = 8), as one would expect of the policy that searches more. With an exact verifier
available at test time, this headroom could be turned into accuracy by sampling several answers and checking them,
which we have not tried.

## 4. Reward exploitation and safeguards

We observed one exploit, early in the project, in small pilot runs where only about 5% of rollouts were correct.
Most groups then contained no correct rollout, and among those a wrong but well-formed answer that had run some code
earned 0.2 more than a truncated attempt. Within four steps the policy learned to submit an answer immediately after
a failed check instead of retrying; retries on held-out problems fell from 33 of 43 to 0 of 41.

We closed the exploit in three ways:
- **Contradicted answers:** the shaping terms are withheld from a final answer that the model's own executed check
  had already shown to be wrong.
- **Invalid numbers:** they are withheld from any wrong answer that uses numbers it was not given.
- **Recipe v2:** groups without a correct rollout are not trained on at all, so the shaping terms can no longer drive
  learning on their own.

The other safeguards are structural:
- The execution term requires an expression over at least two of the given numbers, so printing the target earns
  nothing.
- A rollout that fabricates an `<output>` block ends immediately and receives no shaping reward.
- Training stops itself if, averaged over the last five steps, more than 60% of answers are contradicted or more than
  40% of rollouts are truncated.
- A suite of 20 hand-constructed hacks checks that each one scores below an honest solution. It includes reused or
  invented numbers, printing the target, dummy code, fabricated output, and malformed or duplicated tags.

Partial-solution hints exist only in the training loop, never in evaluation, and the holdout files never reach the
training machines. Tests enforce both.

## 5. Results on the test set

The task description asks for a comparison between base models and trained policies on the original 1,000-problem
test set. Table 3 gives that comparison for every model and training condition. The test set informed some early
decisions in the v1 lineage, so its numbers are slightly optimistic for those runs, and holdout3 remains the clean
basis for comparing models.

*Table 3. Test-set results (1,000 problems, greedy, 768 tokens, two code rounds). "Recovered" counts problems solved
after a first check that did not find the target, out of all such problems. Token counts use the Qwen3 tokenizer for
every model. Rows marked † were evaluated with the HF backend on the workstation, the others with vLLM; on this set
the two backends' greedy outputs differ by about one point for the same checkpoint.*

| Model | Training | Prompt | Pass@1 | Easy / Med. / Hard | Format | Tokens | Capped | Recovered |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | none | `zero` | 0.0% | 0 / 0 / 0 | 63.4% | 293 | 17.9% | 0 / 823 |
| Qwen3-4B | none | `r1zero_tool` † | 6.3% | 48 / 10 / 5 | 61.1% | 495 | 38.3% | 25 / 809 |
| Qwen3-4B | none | demo | 11.3% | 85 / 23 / 5 | 17.6% | 612 | 82.3% | 8 / 722 |
| Qwen3-4B | v1, seed 1 | demo | 53.0% | 339 / 107 / 84 | 99.5% | 301 | 0.5% | 81 / 540 |
| Qwen3-4B | v1, seed 2 | demo | 50.9% | 329 / 109 / 71 | 92.5% | 383 | 7.5% | 66 / 536 |
| Qwen3-4B | v1, hand-tuned | demo | 52.7% | 331 / 116 / 80 | 99.7% | 302 | 0.3% | 67 / 531 |
| Qwen3-4B | v2, seed 2 | demo | 70.0% | 358 / 174 / 168 | 91.7% | 341 | 8.3% | 320 / 608 |
| Qwen3-4B | v2, seed 2 | `r1zero_tool` † | 70.7% | 357 / 176 / 174 | 97.4% | 312 | 2.6% | 52 / 253 |
| Qwen3-8B | none | demo | 11.8% | 102 / 11 / 5 | 0.1% | 645 | 74.7% | 13 / 270 |
| Qwen3-8B | v1 | demo | 58.5% | 348 / 134 / 103 | 98.8% | 314 | 1.2% | 86 / 494 |
| Gemma-4-E4B | none | demo | 4.6% | 38 / 4 / 4 | 83.3% | 287 | 16.7% | 0 / 899 |
| Gemma-4-E4B | none | `r1zero_tool` † | 2.0% | 19 / 1 / 0 | 33.7% | 549 | 66.3% | 1 / 142 |
| Gemma-4-E4B | v1 | demo | 31.7% | 252 / 44 / 21 | 98.4% | 367 | 1.6% | 41 / 711 |
| Gemma-4-E4B | v2 | `r1zero_tool` † | 63.6% | 350 / 148 / 138 | 99.9% | 284 | 0.0% | 9 / 315 |
| Gemma-4-E4B | v2, +100 steps | `r1zero_tool` † | 70.2% | 368 / 176 / 158 | 99.9% | 298 | 0.1% | 9 / 300 |

The base models fail in characteristic ways:
- **Base Qwen3-4B with `zero`** produces well-formed output but answers with the target itself.
- **The base models with `r1zero_tool`** run code readily (Qwen on 87% of problems) but rarely to any purpose, and
  often run out of budget: 38% of Qwen's and 66% of Gemma's responses are truncated.
- **The Qwen base models with the demo prompt** mostly run out of budget: 82% and 75% of their responses are
  truncated, which also explains their very low format compliance.

Reinforcement learning removes both failure modes:
- **Shorter answers:** the trained policies are shorter than the demo-prompted base models (roughly 300–380 tokens
  against more than 600), and nearly all of them comply with the format.
- **Recovery after a miss:** under v1, the trained policies solve 12–17% of problems after a failed first check,
  against at most 5% for the bases. Recipe v2 raises this to 53% with the demo prompt. The zero-shot policy recovers
  less often (21%) but misses far less often in the first place, 253 times against 608. Gemma almost never recovers
  (9 of 300 after the continuation): once its first check misses, the answer is almost always wrong.

## 6. Compute

Generation dominates the cost of a GRPO step, so most of the engineering went into rollouts:
- **Colocated vLLM:** vLLM runs in the training process with sleep mode, which offloads its weights and frees the KV
  cache during the backward pass. A full step of 512 rollouts takes about 2.3 minutes on one RTX PRO 6000 (96 GB).
- **Chunked log-probabilities:** they are computed in chunks over the vocabulary, so the full
  sequence × vocabulary tensor is never materialised.
- **Evaluation on the workstation:** we wrote a grouped-query attention kernel that avoids replicating the KV cache
  and merged the LoRA weights for decoding, which made Hugging Face generation about ten times faster on the AMD
  workstation. All reported evaluations nonetheless keep the adapter unmerged. The analyses in Section 3 also used a
  second local GPU (RTX A4000) with the same code and lock file.
- **Preemption and cost:** all training used spot instances. Checkpoints are written atomically to cloud storage every
  two steps, runs resume automatically after preemption, each run carries a cost cap, and a lineage powers its VM
  off when it finishes.

A complete 300-step lineage costs 10–16 GPU-hours on the 4B-class models. The zero-shot Qwen3-4B run took 11.9.
Everything reported here, including pilots, ablations, continuations and runs stopped by a guard, used 144 GPU-hours
of training, about $290 at the spot price of roughly $2 per hour. The figures are measured from the checkpoint
heartbeats by `scripts/cost_report.py`. All evaluations ran locally at no cost.

## 7. Limitations and deviations from the task description

**Statistical power and seeds.** The central comparison rests on one completed seed per arm, the same seed for both.
The three v1 lineages of Qwen3-4B suggest that run-to-run variation on holdout3 is about 2 points. That is small next
to the 16–18-point gain from recipe v2, but of the same order as the 1.8-point difference between the zero-shot and
demo arms. The estimate also comes from recipe v1, and v2 almost certainly varies more. It has no KL penalty, and
seed 1 was stopped by the length guard under both prompts while seed 2 finished under both. The two v2 arms are
therefore matched in their sensitivity to the seed, but we have no estimate of v2's variance at the end of training.
Our claim of parity between the arms is the most the data support, and any claim that one is better would need
several more completed seeds, which in turn probably needs a guard threshold, or a length penalty, set so that
seed-to-seed differences do not decide whether a run finishes. The recipe v2 comparison itself was registered for two
seeds, so its registered two-seed average could not be computed. Holdout3 has now been used to compare more than a dozen
checkpoints. Each checkpoint was evaluated only once and none was selected on it, but the individual p-values are not
corrected for multiple comparisons.

**What "emergent" means here.** The zero-shot prompt contains no example and no strategy, but it does tell the model
that the tool exists and what it returns. Using the tool is therefore instructed. What the model discovers is how to
use it: when to check, what to print, and how to respond to a miss. In the demo arms, retrying after a failed check is
also part of the instruction. The base model already retries in most cases, and RL mainly improves the quality of the
first check and of the retry. The trained policies are also closely tied to their prompts: the zero-shot policy loses
more than half its accuracy under the demo prompt. Rewordings that keep the content cost it between nothing and 3
points (Section 2.2). In the zero-shot arms, the tool use that emerged is mostly verification of an answer derived in
the reasoning (Section 3); the policies would solve most problems without the tool. We regard that as an honest
reading of what the reward asked for, since nothing in it favours solving with code over solving without it.

**Comparability across models.** We kept hyperparameters identical across models on purpose, to test the recipe
rather than each model's best achievable score. The Gemma results are therefore not tuned. The comparison between
Gemma under v1 and under v2 changes both the recipe and the prompt, so it shows what the configuration achieves but
does not attribute the gain to either change. Gemma-4 was run as a text-only model extracted from a multimodal
checkpoint. The extracted model's logits match the full model exactly, but it is not an officially released variant.

**Evaluation details.** Greedy decoding with batched LoRA kernels in vLLM is not bit-reproducible: repeated passes over
300 validation problems differ by up to seven problems. We therefore averaged three passes for every
validation-based decision, and ran all holdout evaluations with the HF backend. The 768-token budget truncates 4% of
the final zero-shot policy's holdout3 responses and 10% of the demo policy's. Doubling it to 1,536 tokens changes
neither score significantly (69.5% and 69.0%, against 69.9% and 68.1%; p = 0.77 and 0.46). The demo policy gains 16
hard problems with the longer budget but loses some elsewhere, so the budget costs it at most a point. Token counts
for Gemma in Table 3 use the Qwen3 tokenizer and are comparable only approximately.

**Missing log steps.** Until it was fixed, a bug in resuming after preemption overwrote the beginning of the training
log. As a result, Fig. 2 has gaps at:
- steps 0–29 of phase b and 0–11 of phase c of the v2 demo run;
- steps 74–75 of phase d of the zero-shot run.

Steps 0–13 of phase b of the stopped v2 demo seed are also missing. Checkpoints, validation passes and evaluations
were unaffected, and the bug is fixed.

**Deviations from the task description.**
- **Main model:** we use Qwen3-4B-Base rather than gemma-2-2b. The reasons are in Section 2.3, and results for the
  Gemma-4 models are reported alongside.
- **No supervised cold start:** we tried programmatic SFT traces early on, and they did not improve on the base
  model. The prompt and the signal-based curriculum take their place, and the zero-shot result suggests that a cold
  start is unnecessary once the prompt describes the tool.
- **Sandbox:** it runs in a separate worker process rather than in a thread of the training process, for the
  isolation reasons given in Section 1.2.
