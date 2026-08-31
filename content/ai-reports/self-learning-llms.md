# The Landscape of Self-Learning LLMs: Self-Improvement, Self-Play, and Data-Efficient Training

## TL;DR

- "Self-learning" LLMs are real and advancing fast, but the term conflates six distinct paradigms; the most convincing results (DeepSeek-R1-Zero, Absolute Zero Reasoner, R-Zero) come from **reinforcement learning with verifiable rewards (RLVR)** in domains where correctness is cheaply checkable (math, code), and even the most aggressive "zero data" systems presuppose a large pretrained base model — no LLM learns from literally nothing.
- The single biggest open debate is whether self-improvement **creates** new capability or merely **elicits** latent capability already in the base model; strong pass@k evidence (Yue et al., 2025) shows base models consistently surpass RL-trained models as k increases and that RLVR reasoning paths "are already included in the base models' sampling distribution," suggesting current RLVR mostly sharpens existing abilities rather than expanding the reasoning boundary — though this is contested.
- On the data-efficiency branch, sample-efficient training (BabyLM, TinyStories, the phi/"textbooks" line) and synthetic-data pipelines (Nemotron-4, phi-4) show you can go very far with tiny or fully synthetic corpora — but they lean on curation and, usually, a stronger teacher model, and training naively on self-generated data risks **model collapse**.

## Key Findings

- Self-play RLVR with zero human-curated data works in verifiable domains: Absolute Zero Reasoner and R-Zero both improve reasoning with no external dataset, but both start from a capable pretrained base and a verifier (code executor / majority vote).
- DeepSeek-R1-Zero demonstrated that reasoning behaviors ("aha moments," self-verification, long chains of thought) emerge from pure RL on a base model with rule-based rewards, no SFT cold start.
- The pass@k critique is the central empirical check on hype: RLVR improves pass@1 sampling efficiency but base models often match or exceed RLVR models at large k.
- Data efficiency: BabyLM shows sub-100M-word models can beat models trained on trillions of words on targeted benchmarks; phi-4 (14B) uses synthetic data as the bulk of pretraining and significantly exceeds its teacher GPT-4o on GPQA (56.1 vs 50.6) and MATH (80.4 vs 74.6).
- Model collapse (Shumailov et al., Nature 2024) is the key risk for naive self-training; mitigations include mixing real data, verification/filtering, and accumulate-don't-replace strategies.

## Taxonomy: Six Senses of "Self-Learning"

The phrase "self-learning LLM" is used loosely. It helps to separate six distinct things:

1. **Self-supervised pretraining** — the original "self-learning": next-token prediction on raw text with no human labels. This is the foundation of every LLM and is what makes all the other senses possible. It is "self-supervised," not "self-improving."
2. **Self-training / pseudo-labeling** — a model labels unlabeled data, filters, and retrains on its own high-confidence outputs (STaR, ReST, RFT). Classic semi-supervised learning applied to LLMs.
3. **Self-play** — a model (or two copies) generate tasks and/or responses for each other and co-evolve (SPIN, SPPO, Absolute Zero, R-Zero). Inspired by AlphaZero.
4. **Self-rewarding** — a model judges its own outputs and uses those judgments as the training signal (Self-Rewarding LMs, Meta-Rewarding, Constitutional AI/RLAIF).
5. **Self-adapting / test-time** — a model updates its own weights or behavior at inference/deployment time (SEAL, Transformer², test-time training).
6. **Autonomous / agentic learning** — an agent learns from its own interaction trajectories, memory, and self-modification (Era of Experience, Darwin Gödel Machine, AI Scientist).

A crucial honest-accounting point: every "zero data" system today lives in sense 2–6 and **presupposes sense 1**. The "zero" refers to zero human-curated _task/label_ data for the post-training stage, not zero data overall.

---

## BRANCH A — Self-Improving / Self-Training LLMs

### A1. Self-generated reasoning bootstrapping (STaR and descendants)

The foundational method is **STaR (Self-Taught Reasoner)**, Zelikman, Wu, Mu & Goodman, Stanford, NeurIPS 2022 (arXiv 2203.14465). The loop: generate chain-of-thought rationales for questions with few-shot prompting; keep rationales that yield the correct answer; for wrong answers, "rationalize" by giving the model the correct answer and having it produce a justifying rationale; fine-tune on all successful rationales; repeat. On CommonsenseQA, STaR-with-rationalization (base model: 6B GPT-J) reached 72.5% on the dev set — far closer to the 73.0% of the 30× larger GPT-3 than to few-shot baselines, and well above few-shot 137B LaMDA's 55.6%.

Descendants:

- **Quiet-STaR** (Zelikman et al., 2024, arXiv 2403.09629) generalizes STaR so the model generates token-wise rationales ("think before speaking"); zero-shot gains on GSM8K (5.9%→10.9%) and CommonsenseQA (36.3%→47.2%).
- **V-STaR** (Hosseini et al., Mila/DeepMind, 2024, arXiv 2402.06457) trains a verifier via DPO on both correct and incorrect generated solutions, used at inference to select among candidates; 4–17% test accuracy improvement over prior self-improvement approaches on math and code with LLaMA2.
- **ReST** (Gulcehre et al., Google DeepMind, 2023, arXiv 2308.08998) — "Reinforced Self-Training," a growing-batch RL method with an inner "Improve" loop and outer "Grow" loop; the original paper evaluated on machine translation (not MATH/APPS — a common confusion).
- **ReST-EM / "Beyond Human Data"** (Singh et al., DeepMind, 2023/24, arXiv 2312.06585, TMLR 2024) — EM-style self-training: generate, filter by binary reward, fine-tune, repeat; tested on MATH and APPS with PaLM-2, scaling favorably with model size and significantly surpassing fine-tuning on human data.
- **RFT (rejection sampling fine-tuning)** — the simplest widely-used version: sample many solutions, keep correct ones, SFT on them. A component of the DeepSeek-R1 and Llama post-training pipelines.

### A2. Self-rewarding and LLM-as-a-judge bootstrapping

- **Self-Rewarding Language Models** (Yuan et al., Meta AI, 2024) — the model plays LLM-as-a-judge on its own outputs, creating preference pairs it trains on via iterative DPO. The reward model is not frozen; it improves alongside the policy.
- **Meta-Rewarding LMs** (Wu, Yuan, Golovneva, Xu, Tian, Jiao, Weston, Sukhbaatar, Meta, 2024, arXiv 2407.19594, EMNLP 2025) — adds a meta-judge that judges the model's own judgments, improving judgment quality. Llama-3-8B-Instruct win rate on AlpacaEval 2 improved from 22.9% to 39.4%, and on Arena-Hard from 20.6% to 29.1%, with no human supervision. The authors note that judgment-only self-rewarding saturates rapidly, which motivated the meta-judge.
- **Constitutional AI / RLAIF** (Bai et al., Anthropic, 2022, arXiv 2212.08073) — a harmless assistant trained via self-critique/revision (supervised stage) and RL from AI feedback against a preference model trained on AI-generated preferences, with only a short list of human-written principles ("constitution") as human input. Follow-up work (Constitution or Collapse?, 2024) found small models can suffer collapse under CAI self-critique, while larger models tolerate it.

### A3. Self-play with little/no human data

- **SPIN (Self-Play Fine-Tuning)** (Chen et al., UCLA, ICML 2024, arXiv 2401.01335) — starts from an SFT model; the model generates responses and is trained (DPO-style) to distinguish its own generations from human SFT responses, iteratively. Improved a zephyr-7b SFT model from 58.14% to 63.16% on the HuggingFace Open LLM average. Requires an existing SFT dataset.
- **SPPO (Self-Play Preference Optimization)** (Wu, Sun, Yuan et al., UCLA/CMU, 2024, arXiv 2405.00675, ICLR 2025) — frames alignment as a two-player constant-sum game seeking a Nash equilibrium. Using 60k UltraFeedback prompts and a 0.4B PairRM preference model, fine-tuned Mistral-7B-Instruct-v0.2 to a 28.53% length-controlled win rate vs GPT-4-Turbo on AlpacaEval 2.0; from Llama-3-8B-Instruct, 38.77%.
- **Absolute Zero / Absolute Zero Reasoner (AZR)** (Zhao et al., 2025, arXiv 2505.03335, NeurIPS 2025) — a single LLM acts as both proposer (generates coding tasks spanning abduction, deduction, induction) and solver; a code executor provides verifiable rewards. Trained with zero external data, AZR achieves overall SOTA on coding and math reasoning, outperforming zero-setting models trained on tens of thousands of human-curated examples. Still requires a capable base model (experiments on Qwen2.5-Coder) and a code executor as its grounding.
- **R-Zero** (Huang et al., 2025, arXiv 2508.05004, ICLR 2026) — initializes a base model into a Challenger and a Solver that co-evolve: the Challenger is rewarded for posing tasks at the edge of the Solver's ability, the Solver for solving them. Qwen3-4B-Base gained +6.49 points on math benchmarks after three iterations; gains transfer to general-domain reasoning (+3.81 on Qwen3-8B-Base, +3.65 on OctoThinker-3B). Uses majority-vote pseudo-labels (no ground truth).
- **Genius** (Xu et al., 2025, arXiv 2504.08672, ACL 2025) — a purely unsupervised self-training framework using stepwise foresight re-sampling and an advantage-calibrated optimization (ACO) loss, with no external reward model or verifier; trained on ~25K unsupervised general queries (Magpie/OpenHermes), boosting reasoning ~7% on average and by 6.67% in a harder scenario.

### A4. Reinforcement learning with verifiable rewards (RLVR) and reasoning models

RLVR is the engine behind the 2025 reasoning-model wave. Instead of a learned reward model (hackable), the reward is a rule-based check: is the math answer correct, do the unit tests pass?

- **DeepSeek-R1 / R1-Zero** (DeepSeek-AI, 2025, arXiv 2501.12948; Nature, Sept 2025, doi 10.1038/s41586-025-09422-z) — R1-Zero was trained directly on DeepSeek-V3-Base with GRPO (Group Relative Policy Optimization) and rule-based accuracy+format rewards, no SFT cold start. It naturally developed long chains of thought, self-verification, and the famous "aha moment," and its response length grew over training. R1 added a small cold-start dataset to fix readability and language-mixing, achieving performance comparable to OpenAI o1-1217 on reasoning. Corresponding author Liang Wenfeng; the paper made Nature's cover. Six distilled dense models (1.5B–70B, Qwen/Llama) were open-sourced.
- **Kimi k1.5** (Moonshot AI, 2025), **OpenAI o-series** (o1/o3, training details undisclosed), and **Qwen QwQ** are the other main RLVR reasoning families.
- **The elicitation-vs-creation debate**: "Does RLVR Really Incentivize Reasoning Capacity Beyond the Base Model?" (Yue, Chen, Lu, Zhao et al., LeapLab — Tsinghua & Shanghai Jiao Tong University, 2025, arXiv 2504.13837, NeurIPS 2025) found that while RLVR models beat base models at pass@1, **base models consistently surpass RL-trained models across all benchmarks and LLM families as k increases** (e.g., k=256). Over training, training-set pass@1 rose from 26.1% to 42.5% while pass@256 steadily fell. Their verbatim conclusion: "the reasoning paths generated by RLVR models are already included in the base models' sampling distribution, suggesting that their reasoning abilities originate from and are bounded by the base model." They also found distillation (unlike RL) can introduce genuinely new knowledge. This is contested: later work (curriculum RL, and analyses of overtraining/diversity-collapse, 2025–2026) argues aggregate pass@256 decline is partly an overtraining artifact and that longer/curriculum RL can expand the boundary on problems the base model cannot solve at all (Pass@256 = 0).

### A5. Test-time training and self-adapting models

- **SEAL (Self-Adapting Language Models)** (Zweiger, Pari, Guo, Akyürek, Kim, Agrawal, MIT, 2025, arXiv 2506.10943, NeurIPS 2025) — the model generates its own "self-edits" (natural-language finetuning data + update directives), performs SFT on them (inner loop), and uses RL with downstream performance as the reward to improve the self-edit policy (outer loop). The revised (Sept 2025) version shows self-adaptation scales with model size, integrates RL to reduce catastrophic forgetting, and formalizes the dual-loop structure.
- **Transformer² (Transformer-squared)** (Sun, Cetin, Tang, Sakana AI, 2025, arXiv 2501.06252) — self-adaptive at inference via a two-pass mechanism: identify the task, then mix RL-trained "expert vectors" (z-vectors) that scale the singular values (SVD) of weight matrices via Singular Value Fine-tuning. Outperforms LoRA with fewer parameters; expert vectors transfer across models (Llama3-8B-Instruct → Mistral-7B-Instruct-v0.3).
- **Test-time training (TTT)** more broadly updates parameters on each test instance; strong results on abstraction benchmarks (e.g., ARC) when combined with augmentation.

### A6. Continual/lifelong and experiential learning

- **"The Era of Experience"** (David Silver & Richard Sutton, DeepMind, 2025, preprint chapter for the MIT Press book _Designing an Intelligence_) argues AI is moving from the "Era of Human Data" to an "Era of Experience," where agents learn primarily from self-generated experience via interaction and grounded rewards, citing AlphaProof (2024) as an early marker. Their thesis: "any static procedure for synthetically generating data will quickly become outstripped" — data must be generated by agents interacting with an environment and improving continually. It is a position paper/manifesto, not an empirical result, and has drawn alignment critiques (e.g., that reward-driven experiential agents pose unsolved alignment problems).

### A7. Recursive self-improvement and AI-generating-AI

- **Darwin Gödel Machine (DGM)** (Zhang, Hu, Lu, Lange, Clune, Sakana AI/UBC/Vector Institute, 2025, arXiv 2505.22954) — a self-improving _coding agent_ that rewrites its own code, keeping an expanding archive of variants (Darwinian open-ended evolution) validated on coding benchmarks. It improved itself from 20.0% to 50.0% on SWE-bench and from 14.2% to 30.7% on Polyglot, discovering emergent improvements (patch validation, better editing tools). Critically, the foundation model itself is _frozen_; what evolves is the agent's code/scaffolding.
- **The AI Scientist** (Lu, Lu, Lange, Foerster, Clune, Ha, Sakana AI/Oxford/UBC, 2024, arXiv 2408.06292) — an end-to-end pipeline that generates ideas, writes/runs code, and produces full papers at less than ~$15/paper. **AI Scientist-v2** (Yamada et al., 2025, arXiv 2504.08066) produced a paper that passed peer review at the ICLR 2025 "I Can't Believe It's Not Better" workshop (average reviewer score 6.33), the first fully AI-generated manuscript to pass peer review — though Sakana disclosed that none of the three submitted papers met their internal bar for a main-conference paper and the accepted one was to be withdrawn.
- **Gödel Agent** (Yin et al., ACL 2025) and **Promptbreeder** (Fernando et al., ICML 2024) are related self-referential self-improvement frameworks.

### A8. Model collapse: the fundamental risk of self-training

**Shumailov, Shumaylov, Zhao, Papernot, Anderson & Gal, "AI models collapse when trained on recursively generated data," Nature 2024 (631:755–759, doi 10.1038/s41586-024-07566-y)** showed that indiscriminately training successive generations of models on their predecessors' outputs causes irreversible degradation — the tails of the distribution disappear first, then the distribution converges to a low-variance point estimate. The effect appears in LLMs, VAEs, and Gaussian mixtures. Follow-up analysis (Borji, 2024, arXiv 2410.12954) argues the phenomenon is a fundamental statistical property of repeated re-fitting. Mitigations found in later work: keep a fraction of real data, **accumulate rather than replace** data across generations, use verification/filtering (the AZR/R-Zero verifier approach), and machine-generated-text detection (arXiv 2502.15654). Model collapse is why "self-play works, so let's train on synthetic forever" is naive: it only works cleanly where a **verifier or real anchor** keeps the process grounded.

---

## BRANCH B — LLMs Trained on Little or No External Data

### B1. Sample-efficient pretraining: the BabyLM Challenge

The **BabyLM Challenge** (Warstadt, Mueller, Choshen, Wilcox, Zhuang, et al., first edition CoNLL 2023; annual since) restricts pretraining to a developmentally plausible budget — 10M words (strict-small) or 100M words (strict), roughly what a 13-year-old has heard (2M–7M words/year). Findings (arXiv 2504.08165): winning submissions using the **LTG-BERT** architecture (Samuel et al., 2023) outperformed models trained on trillions of words on the challenge's grammar/understanding benchmarks (BLiMP, GLUE, EWoK). Later editions (2024) added vision-language tracks and found data augmentation, knowledge distillation, and even RNN-based architectures (HGRN2) competitive with transformers in the 10M/100M tracks. The point: for many linguistic-competence targets, architecture and data curation matter more than raw scale.

### B2. Small models on tiny/synthetic corpora: TinyStories and the phi line

- **TinyStories** (Eldan & Li, Microsoft, 2023, arXiv 2305.07759) — a synthetic dataset of children's stories (GPT-generated, ~3-4-year-old vocabulary); models from 2.5M–80M parameters produce coherent, grammatical English and show interpretable attention heads/neurons.
- **phi-1** ("Textbooks Are All You Need," Gunasekar et al., 2023, arXiv 2306.11644) — 1.3B params, trained on "textbook-quality" synthetic + filtered code data (trained in 4 days on 8 A100s); Python coding near-SOTA on HumanEval despite tiny size.
- **phi-1.5** (Li et al., 2023, arXiv 2309.05463) — 1.3B, common-sense reasoning comparable to models 5× larger; notably, the "absence of web data" reduced toxicity.
- **phi-3-mini** (Microsoft, 2024, arXiv 2404.14219) — 3.8B params, 3.3T tokens, 69% MMLU / 8.38 MT-bench, deployable on a phone; phi-3-small/medium (7B/14B, 4.8T tokens) reach 75%/78% MMLU and 8.7/8.9 MT-bench. The innovation "lies entirely in our dataset" — filtered web + synthetic.
- **phi-4** (Microsoft, Dec 2024, arXiv 2412.08905) — 14B, trained on ~9.8T tokens where synthetic data "constitutes the bulk of the training data." It significantly exceeds its teacher GPT-4o on GPQA (56.1 vs 50.6) and MATH (80.4 vs 74.6), with MMLU 84.8 and HumanEval 82.6, and meets or exceeds Llama-3.1-405B on reasoning benchmarks. Ablations showed >50% synthetic data early in pretraining improves reasoning-heavy benchmarks.
- **Cosmopedia** (Hugging Face, 2024) — an open replication of the phi synthetic-textbook approach (~25B tokens generated by Mixtral).

### B3. Fully synthetic-data pretraining pipelines

- **Nemotron-4 340B** (NVIDIA, 2024) — a base/instruct/reward model family explicitly built for synthetic data generation (SDG): the Instruct model generates data, the Reward model (92.2 on RewardBench) ranks/filters it. The base model was trained on 9T tokens; a uniquely permissive open license lets developers train other models on its outputs.
- **Nemotron-CC** and the **Llama-3 synthetic pipelines** apply distillation-heavy and synthetic-rewriting approaches at web scale.
- **Synthetic continued pretraining / EntiGraph** (2024, arXiv 2409.07431) extracts more knowledge from small corpora by generating synthetic elaborations — a route to learning from proprietary or tail-knowledge datasets.

### B4. Data-constrained scaling laws

- **"Scaling Data-Constrained Language Models"** (Muennighoff, Rush, Barak, Le Scao, et al., NeurIPS 2023, arXiv 2305.16264) — with a fixed compute budget and limited data, **up to ~4 epochs of repeated data is nearly as good as fresh data**; beyond that, returns to repetition (and to added compute) decay to zero. Experiments ranged to 900B tokens and 9B params; this is the empirical backbone for the "data wall" discussion.
- **"Will we run out of data?"** (Villalobos, Ho, Sevilla, Besiroglu, Heim, Hobbhahn, Epoch AI, ICML 2024, arXiv 2211.04325) — estimates the "effective stock of quality and repetition adjusted human-generated public text for AI training at around 300 trillion tokens" (90% interval 100T–1,000T; median near 2028) and projects models will fully utilize it between **2026 and 2032**, earlier if intensely overtrained. Proposed continuations: synthetic data, transfer learning, and data-efficiency gains.

### B5. Low-resource languages and cross-lingual bootstrapping

Transfer learning from high-resource to low-resource languages, cross-lingual alignment, and synthetic translation/back-translation remain the main levers where native data is scarce; these are "little data" in the target language but lean heavily on multilingual pretraining and stronger-model distillation.

### B6. Are truly "zero-data" LLMs possible?

No — not in any strong sense. Every current "zero data" or "from scratch" claim (AZR, R-Zero, R1-Zero) means **zero human-curated task/label data for the RL/post-training stage**. All of them start from a base model pretrained on trillions of tokens of human text, and all of them require a **grounding signal**: a code executor, a math checker, or majority-vote consistency. The honest framing: these systems remove the human-annotation bottleneck for _post-training_, and they relocate the "data" into (a) the pretrained base and (b) the verifier/environment. AlphaZero-style tabula-rasa learning works for LLMs only where a perfect, cheap verifier substitutes for the rules of a game.

---

## Named Groups and People

- **DeepSeek** (Liang Wenfeng) — R1/R1-Zero, GRPO, RLVR at scale.
- **Meta AI / FAIR** (Jason Weston, Weizhe Yuan, Sainbayar Sukhbaatar, Tianhao Wu, Yuandong Tian) — Self-Rewarding and Meta-Rewarding LMs.
- **Anthropic** — Constitutional AI / RLAIF.
- **Google DeepMind** (Rishabh Agarwal, Caglar Gulcehre, Avi Singh; David Silver & Richard Sutton) — ReST/ReST-EM, Era of Experience.
- **MIT** (Pulkit Agrawal, Yoon Kim, Ekin Akyürek, Adam Zweiger, Jyothish Pari, Han Guo) — SEAL.
- **Sakana AI** (David Ha, Robert Lange, Chris Lu, Cong Lu, Qi Sun, Edoardo Cetin, Yujin Tang; with Jeff Clune and Jenny Zhang at UBC) — Transformer², Darwin Gödel Machine, AI Scientist.
- **Stanford** (Eric Zelikman, Noah Goodman, Yuhuai Wu) — STaR, Quiet-STaR.
- **UCLA/CMU** (Quanquan Gu, Zixiang Chen, Yue Wu, Zhiqing Sun, Yiming Yang) — SPIN, SPPO.
- **Microsoft Research** (Sébastien Bubeck, Ronen Eldan, Yuanzhi Li, Suriya Gunasekar, Marah Abdin) — TinyStories, phi line.
- **AZR** (Andrew Zhao et al.); **R-Zero** (Chengsong Huang et al.); **Genius** (Fangzhi Xu et al.).
- **Tsinghua/SJTU LeapLab** (Yang Yue, Zhiqi Chen, Rui Lu) — the RLVR pass@k critique.
- **Oxford/Cambridge/Toronto** (Ilia Shumailov, Yarin Gal, Ross Anderson, Nicolas Papernot) — model collapse.
- **Epoch AI** (Pablo Villalobos, Jaime Sevilla, Tamay Besiroglu) — data-wall forecasting.
- **Hugging Face / BigScience** (Niklas Muennighoff, Thomas Wolf, Colin Raffel) — data-constrained scaling laws.

## Open Problems, Limitations, and Hype vs Reality

- **Verifier dependence**: self-play/RLVR shines only where verification is cheap and reliable — code (unit tests) and math (answer checking). Open-ended domains (essays, strategy, science ideation) lack cheap verifiers, so self-improvement is far weaker there.
- **Reward hacking**: learned reward models are hackable; rule-based verifiers are the reason AZR/R1-Zero avoid the worst of this, but verifiers themselves can be gamed (reward from format, degenerate solutions).
- **Elicitation vs creation**: the pass@k evidence (Yue et al.) is the strongest reason for skepticism that RLVR is a route to superhuman capability from a fixed base; distillation may be the only reliable way to inject genuinely new knowledge. The rebuttal (curriculum/longer RL) is credible but not yet decisive.
- **Model collapse and plateaus**: naive recursive training degrades irreversibly; self-training plateaus without fresh grounding (Meta-Rewarding noted rapid saturation of the judge; Constitutional AI collapses on small models).
- **Narrow-domain limits**: nearly all headline "zero data" wins are in math/code; generalization to general reasoning is claimed (R-Zero transfer, +3.65–3.81 points) but modest.
- **Catastrophic forgetting** in continual/self-adapting setups (SEAL explicitly targets this).
- **Recursive self-improvement is scaffolding-level, not weight-level**: DGM and AI Scientist evolve _code/agents around frozen models_, not the models' own weights — the "recursively self-improving superintelligence" framing is not what these systems demonstrate.

## Where the Field Is Heading (2026 and What to Watch)

- **Better verifiers for open-ended domains** — the key bottleneck; watch generative/LLM verifiers, process reward models, and self-consistency signals.
- **Curriculum self-generation** (R-Zero, AZR) maturing into general-purpose self-curricula, now extending to multimodal (MM-Zero) and tool-use (Tool-R0) settings.
- **Test-time/continual learning** moving from proof-of-concept (SEAL, Transformer²) toward deployed agents that learn from their own trajectories.
- **The data wall** (2026–2032 per Epoch, median ~2028) driving synthetic-data and data-efficiency research; the open question is whether synthetic pipelines avoid collapse at scale.
- **The empirical resolution of elicitation-vs-creation** — curriculum RL and longer-horizon RL papers in 2025–2026 are directly testing whether RL can expand the reasoning boundary.
- **Recursive agent self-improvement** (DGM lineage, Red Queen / co-evolving evaluators) and automated ML research as a growing — and safety-relevant — area.

## Recommendations

- **If you need reliable self-improvement now**: use RLVR/RFT in verifiable domains (math, code). Build or buy a solid verifier first; that is the real asset. Benchmark with pass@1 _and_ pass@k — if base-model pass@k at large k already matches your RL model, you are eliciting, not expanding, and should weigh distillation instead.
- **If you are data-constrained**: prefer curation + synthetic data (phi/Cosmopedia recipes) and up to ~4 epochs of repetition (Muennighoff); mix real and synthetic to avoid collapse; use a stronger teacher for distillation where licensing allows (Nemotron-4 for permissive SDG).
- **If you are exploring self-rewarding/self-play**: expect saturation; add a meta-judge or periodic real-data/verifier anchoring; monitor diversity metrics (entropy, pass@k) as early collapse warnings.
- **For continual/test-time adaptation**: SEAL and Transformer² are the reference designs; budget for catastrophic-forgetting mitigation.
- **Thresholds that change the plan**: if pass@k gains persist at large k after long RL → creation is happening, invest more in RL; if diversity/entropy collapses → back off epochs, add real data; if a cheap, reliable verifier exists for your domain → self-play is worth it, otherwise stay with distillation/curated synthetic.

## Caveats

- Several results come from lab blogs, secondary explainers, or preprints not yet peer-reviewed; where a claim rests on a single secondary source (e.g., AI Scientist-v2 workshop-acceptance details, some benchmark deltas) it should be verified against the primary PDF.
- "Zero data" is a marketing-adjacent term; every such system depends on a pretrained base and a verifier/environment.
- The elicitation-vs-creation debate is genuinely unresolved; both the Yue et al. skeptical result and the curriculum-RL optimistic rebuttals are credible and domain-dependent.
- Benchmark numbers are point-in-time and often on different base models; cross-paper comparisons are only roughly commensurable.

## References (primary sources)

- STaR — arXiv 2203.14465
- Quiet-STaR — arXiv 2403.09629
- V-STaR — arXiv 2402.06457
- ReST — arXiv 2308.08998
- ReST-EM / Beyond Human Data — arXiv 2312.06585
- Self-Rewarding LMs — Yuan et al., Meta, 2024
- Meta-Rewarding LMs — arXiv 2407.19594
- Constitutional AI — arXiv 2212.08073
- SPIN — arXiv 2401.01335
- SPPO — arXiv 2405.00675
- Absolute Zero / AZR — arXiv 2505.03335
- R-Zero — arXiv 2508.05004
- Genius — arXiv 2504.08672
- DeepSeek-R1 — arXiv 2501.12948; Nature doi 10.1038/s41586-025-09422-z
- Does RLVR Incentivize Reasoning Beyond Base Model — arXiv 2504.13837
- SEAL — arXiv 2506.10943
- Transformer² — arXiv 2501.06252
- Era of Experience — Silver & Sutton, 2025 (MIT Press preprint)
- Darwin Gödel Machine — arXiv 2505.22954
- The AI Scientist — arXiv 2408.06292; v2 — arXiv 2504.08066
- Model collapse — Nature doi 10.1038/s41586-024-07566-y; note — arXiv 2410.12954
- TinyStories — arXiv 2305.07759
- phi-1 / Textbooks Are All You Need — arXiv 2306.11644
- phi-1.5 — arXiv 2309.05463
- phi-3 — arXiv 2404.14219
- phi-4 — arXiv 2412.08905
- BabyLM Findings — arXiv 2504.08165
- Scaling Data-Constrained LMs — arXiv 2305.16264
- Will We Run Out of Data — arXiv 2211.04325
- Synthetic continued pretraining / EntiGraph — arXiv 2409.07431