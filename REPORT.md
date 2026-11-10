# Tool-Use Reasoning from a Tool Description: GRPO with LoRA on Countdown

*This is the two-page technical memo. The full paper is [REPORT_FULL.md](REPORT_FULL.md), and the record of every run,
with the decision rule registered before each result, is [EXPERIMENTS.md](EXPERIMENTS.md).*

**Choice of model.** The task description suggests gemma-2-2b, so the first thing we measured was whether that model
gives reinforcement learning anything to start from. GRPO computes advantages within a group of rollouts for the same
problem, and it therefore receives a gradient only from groups in which some rollouts succeed and others fail. We
sampled each candidate base model eight times on 256 training problems and counted the problems it solved at least
once: gemma-2-2b solved none of them, Gemma-4-E2B solved 1, Gemma-4-E4B solved 8 and Qwen3-4B-Base solved 58. For this
reason Qwen3-4B-Base is our main model, and we transfer the recipe unchanged to Gemma-4-E4B.

**Method.** We train LoRA adapters of rank 16 with GRPO, using 64 prompts and 8 rollouts per prompt in each of 300
steps. When the model closes a `<python>` block, generation pauses, the code runs in the sandbox, and its standard
output is inserted between `<output>` tags with label −100, so that the policy is never trained to predict the
environment. The reward is R = R_correct + R_exec + R_format − λ · max(0, L − 512), in which an exact verifier awards
1.0 for a correct answer, working code and well-formed tags earn 0.1 each, and λ = 0.001 per token. We compare two
training recipes. Recipe v1 applies an adaptive KL penalty, resamples failed groups with a hint, and trains on every
group, whereas recipe v2 removes the penalty and the hints, trains only on groups that contain a correct rollout, and
clips gradients. We also compare two prompts. The demo prompt contains one worked example, while the zero-shot prompt
only states that code inside `<python>` tags is executed and that its output is returned. Table 1 reports the test set
that the task requests together with holdout3, a set of 1,000 problems generated after every training decision, on
which each final checkpoint was evaluated once.

| Model | Training | Prompt | Test pass@1 | Holdout3 | Format | Tokens | Solved after a failed check |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | none | zero-shot | 6.3% | 6.7% | 61.1% | 495 | 25 of 809 |
| Qwen3-4B | none | demo | 11.3% | 9.4% | 17.6% | 612 | 8 of 722 |
| Qwen3-4B | v1 (seed 1) | demo | 53.0% | 51.2% | 99.5% | 301 | 81 of 540 |
| Qwen3-4B | v2 | demo | 70.0% | 68.1% | 91.7% | 341 | 320 of 608 |
| Qwen3-4B | v2 | zero-shot | **70.7%** | **69.9%** | 97.4% | 312 | 52 of 253 |
| Gemma-4-E4B | none | zero-shot | 2.0% | 1.9% | 33.7% | 549 | 1 of 142 |
| Gemma-4-E4B | v2, 400 steps | zero-shot | 70.2% | 68.1% | 99.9% | 298 | 9 of 300 |

*Table 1. Results on the 1,000-problem test set (greedy decoding, 768 new tokens, at most two code executions) and
pass@1 on holdout3. The last column counts the problems solved among those on which the first code check did not find
the target.*

**Findings.** The training recipe matters more than the size of the model. With the same seed, recipe v2 improves on
recipe v1 by 18.3 points on holdout3 (95% interval 15.1 to 21.5), which is more than Qwen3-8B gains over Qwen3-4B
under v1 (55.7%). A description of the tool is also sufficient for learning: the zero-shot arm matches the arm trained
with a demonstration (69.9% against 68.1%, McNemar p = 0.23), and the two remain level after 45 further steps each. A
prompt that shows the tags without saying that code is executed leaves the base model at 0%. What the policy learns to
do with the tool nevertheless depends on the prompt. When execution is disabled, the zero-shot policy loses 6 points
and the demo policy 37. The zero-shot policy derives the answer in its reasoning and runs code to confirm it, although
it does read the result, because falsified output costs it 11 points. The demo policy instead searches with code, and
it goes on to solve 53% of the problems on which its first check fails. Finally, the recipe transfers to Gemma-4-E4B,
which reaches 60.5% after 300 steps and 68.1% after 100 more, but not to Gemma-4-E2B, whose sampling entropy falls
from 0.77 to 0.10 within 50 steps, so that its groups become identical and it ends at 4.4%.

**Learning dynamics.** Reward and validation accuracy rise together over the whole schedule (Figure 1). The zero-shot
arm starts 11 points behind the demo arm and draws level within about 20 steps. Its responses first lengthen on their
own to about 470 tokens and then shorten to about 350 as its checks become more targeted, and the share of its
rollouts that reach the token budget falls from 26% to 1% under the length penalty. Trained policies are consequently
shorter than the base models, most of whose responses run into the budget (Table 1).

![Figure 1](docs/figures/fig2_dynamics.png)

*Figure 1. Training dynamics of Qwen3-4B under recipe v2 with each prompt.*

**Backtracking.** The following trace of the zero-shot policy on a hard holdout3 problem is verbatim, with whitespace
condensed. The printed value shows the model that its expression does not match its own arithmetic, and it abandons
that decomposition and builds another.

```
numbers 6 7 2 25 1 8, target 168
<think> 6 * 28 = 168. 28 can be made from 25 + 7 - 2 = 30 - 2 = 28 ... </think>
<python> print(6 * (25 + 7 - 2)) </python>  <output> 180 </output>
<think> That didn't work, let's try another combination: 8 * 21 = 168. 21 can be made from 7 * 3,
        and 3 can be 6 - 2 - 1 ... </think>
<python> print(8 * (7 * (6 - 2 - 1))) </python>  <output> 168 </output>
<answer> 8 * (7 * (6 - 2 - 1)) </answer>
```

**Test-time compute.** Accuracy scales with the number of samples but not with the number of tokens. Doubling the
budget to 1,536 tokens changes neither policy significantly (p ≥ 0.46), and a second code execution adds 4 points for
the demo policy and 2 for the zero-shot policy. Because the verifier needs only the problem and no reference solution,
we can instead sample at temperature 0.9 and return the first answer that it accepts. With up to eight samples this
solves 87.6% of the test set for the zero-shot policy and 90.1% for the demo policy, at an average cost of only 2.4
and 2.3 samples per problem, since most problems are solved by the first sample. The base model reaches 20.0% with
eight samples, less than a third of what a trained policy solves with one (67.2%), so training did more than
concentrate probability on answers that the base model already produced.

**Reward exploitation.** We observed one exploit, in pilot runs in which only about 5% of rollouts were correct. A
wrong but well-formed answer that had run some code earned 0.2 more than a truncated attempt, and within four steps
the policy learned to answer immediately after a failed check; retries fell from 33 of 43 problems to 0 of 41. We
closed it by withholding the shaping terms from any answer that the model's own check had contradicted and from any
wrong answer that uses numbers it was not given. In addition, R_exec is paid only for code that combines at least two
of the given numbers, a rollout that fabricates an `<output>` block ends at once, and recipe v2 does not train on
groups without a correct rollout.

**Compute.** Generation dominates the cost of a GRPO step, so most of the engineering went into rollouts. vLLM runs
inside the training process and releases its memory during the backward pass, which brings a step of 512 rollouts to
about 2.3 minutes on one 96 GB GPU. Log-probabilities are computed in chunks over the vocabulary, and atomic
checkpoints every two steps absorb spot preemption. One 300-step lineage costs 10 to 16 GPU-hours.

**Rubric.** The sandbox validates code against an AST whitelist with three builtins and runs it in an isolated,
persistent worker process that is killed after 50 ms, and a 10,000-execution stress test checks it for leaks. Output
tokens are masked and advantages are normalised within each group, and independent tests cover both. We implemented
the adaptive KL penalty, found that its coefficient never left its floor, and removed it in v2; the thresholded length
penalty and a guard that stops a run at 40% truncation control verbosity instead. The curriculum samples each tier in
proportion to its current rate of mixed-outcome groups, which is the quantity GRPO needs and for which entropy is only
a proxy. `scripts/reproduce.sh` rebuilds the environment, the data and the figures from one lock file.

**Limitations and deviations.** Each v2 arm has one completed seed, and a second seed was stopped by the truncation
guard under both prompts, so we claim parity between the prompts and nothing stronger. We deviate from the task
description in three respects: the main model is not gemma-2-2b, the sandbox is a separate process because a thread
cannot be killed, and there is no supervised cold start, which we tried and found to give no gain.
