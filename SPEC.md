Engineering Take-Home Challenge:
Emergent Reasoning via RL &
Programmatic Self-Correction
Role: Data & ML Systems Engineer (LLM Post-Training & Reasoning)
Duration: 5–7 Days (Part-Time, ~15–20 hours total)
Deliverable: Git Repository (code, reproducibility scripts, logs) + Technical Memo
(REPORT.md)
1. Executive Summary & Objective
The objective of this challenge is to demonstrate emergent reasoning and autonomous
programmatic verification in an open-weights base model (e.g., google/gemma-2-2b or
gemma-2-9b using LoRA/QLoRA) via Reinforcement Learning (GRPO or PPO).
Rather than performing supervised fine-tuning on human-written chain-of-thought traces, your
training pipeline must incentivize the policy to:
1. Formulate multi-step mathematical hypotheses inside a <think>...</think> scratchpad.
2. Synthesize executable Python verification snippets inside a <python>...</python> block to
validate interim steps.
3. Inspect injected environment stdout feedback returned within <output>...</output>.
4. Backtrack upon noticing errors or arithmetic discrepancies, arriving at a verified final
solution inside <answer>...</answer>.
2. Task Formulation: The Countdown Numbers Game
Given an input set of numbers and a target integer, the model must output an arithmetic
expression that uses each available input at most once with basic operations (+, -, *, /) to reach
the target exactly.
Rollout Protocol Schema
<think>
Hypothesis: Can we reach 952 by combining (100 + 75) with smaller
factors?
Let's check (100 + 75) * 5.
<python>
target = 952
res = (100 + 75) * 5
print(res == target, res)
</python>
<output>
False 875
</output>
875 is too low. Adjusting approach: test (100 + 19) * 8...
</think>
<answer>
(100 - 3) * 8 + ...
</answer>
Environment Constraints
● Interleaving Protocol: Token generation pauses when the stop token </python> is
generated. The sandbox executes the code block, appends
<output>\n{stdout}\n</output>, and resumes generation.
● Execution Limits: Cap executions at a maximum of 2 code execution rounds per
problem to prevent degenerate infinite loops.
3. Required Deliverables
A. Synthetic Data Generator (data_generator.py)
● Algorithmically generate 10,000 training problems and 1,000 held-out evaluation problems
with guaranteed solvability.
● Partition data into difficulty tiers (e.g., Easy: 4 numbers with 2-digit targets; Hard: 6
numbers with 3-digit targets).
B. Safe In-Process Execution Sandbox (sandbox.py)
● In-process Python sandbox running with an AST whitelist that strictly forbids arbitrary
imports, network calls, and file I/O.
● Enforce thread-level execution timeouts (maximum 50 ms) and truncate stdout to 200
characters.
C. RL Training Pipeline (train_rl.py)
● Implementation of Group Relative Policy Optimization (GRPO) or PPO using Hugging
Face TRL or native PyTorch.
● Group rollout sampling size: G ≥ 4 per prompt.
● Loss Masking: Injected environment tokens inside <output>...</output> must have
loss_mask = 0 (label -100) so the model is not trained to predict environment outputs.
● Composite reward function balancing correctness, code execution validity, and tag
hygiene:
R = Rcorrect + Rexec + Rformat - λ · length
D. Evaluation & Reasoning Metrics (eval.py)
● Benchmark the base pre-trained model zero-shot against your RL-tuned policy on the
1,000-sample test set.
● Evaluate Pass@1 accuracy, format compliance rate, average token length expansion,
and measured frequency of successful self-correction following negative execution
feedback.
E. Technical Memo (REPORT.md)
A 2-page report documenting learning dynamics curves (reward and token length over training
steps), qualitative examples of emergent backtracking, compute bottlenecks, and mitigations
against reward exploitation.
4. Evaluation Rubric
Dimension Senior Expectations (Level 4) Staff / Exceptional (Level 5)
Sandbox & System Safety AST-validated whitelist
rejecting dangerous built-ins;
strict sub-second
process/thread timeout.
Process isolation preventing
memory leaks across 10k
rollouts; clean stdout buffer
management.
RL Formulation & Masking Correct token masking on
environment outputs;
implements GRPO with
group-normalized
advantages.
Implements dynamic
length-penalized advantage
bounds or adaptive KL
regulation to prevent
degenerate verbosity.
Data & Cold-Start Lightweight SFT cold-start or
structured curriculum
preventing gradient
starvation.
Dynamic curriculum scaling
problem difficulty based on
real-time policy entropy and
pass rates.
Analysis & Engineering
Hygiene
Reproducible setup script;
quantitative evidence of
test-time compute scaling.
Detailed diagnosis of failure
modes, reward hacking
behaviors, and scaling
bottlenecks.
5. Reference In-Process AST Sandbox Starter Code
import ast
import contextlib
import io
import multiprocessing
ALLOWED_NODES = {
ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Constant,
ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
ast.FloorDiv, ast.Mod, ast.Pow, ast.Compare, ast.Eq, ast.NotEq,
ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Call, ast.Load, ast.Store
}
def validate_ast(code_str: str) -> bool:
try:
tree = ast.parse(code_str)
for node in ast.walk(tree):
if type(node) not in ALLOWED_NODES:
return False
if isinstance(node, ast.Call):
if not (isinstance(node.func, ast.Name) and
node.func.id in {"print", "range", "len"}):
return False
return True
except SyntaxError:
return False
def _run_worker(code_str: str, queue: multiprocessing.Queue):
stdout_buf = io.StringIO()
safe_globals = {"__builtins__": {"print": print, "range": range,
"len": len}}
try:
with contextlib.redirect_stdout(stdout_buf):
exec(code_str, safe_globals, {})
queue.put((True, stdout_buf.getvalue().strip()[:200]))
except Exception as e:
queue.put((False, f"RuntimeError: {str(e)}"))
def safe_execute(code_str: str, timeout: float = 0.5) -> tuple[bool,
str]:
if not validate_ast(code_str):
return False, "SecurityError: Forbidden syntax or built-in."
q = multiprocessing.Queue()
p = multiprocessing.Process(target=_run_worker, args=(code_str,
q))
p.start()
p.join(timeout)
if p.is_alive():
p.terminate()
p.join()
return False, "TimeoutError: Execution exceeded limit."
if not q.empty():
return q.get()
return False, "ExecutionError: Process failed silently."
