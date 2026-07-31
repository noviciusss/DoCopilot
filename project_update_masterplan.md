# Project Update Masterplan for Placements

This document explains exactly what to change in each project, why the change matters, what it proves in interviews, what can be skipped, and how all of it fits into an AI/ML placement strategy targeting strong campus and off-campus outcomes. It is written as an execution guide, not a theory note.

## Positioning

The portfolio already has enough breadth. The current advantage comes from three differentiated systems: DoCopilot for document RAG, Argus for multi-agent research orchestration, and ContextCore for MCP plus memory. The purpose of updates is not to make the projects look prettier. The purpose is to make each one more defensible, more measurable, and easier to explain under interview pressure.[cite:22][file:17]

The biggest placement risk is not “missing one more AI feature.” It is getting rejected before project discussion because of weak DSA, OA performance, CS fundamentals, or inability to defend project claims precisely. The project work below is therefore intentionally limited and role-specific.[file:1][file:17]

## Portfolio roles

| Project | Final portfolio role | What it should prove after updates | What should not happen |
|---|---|---|---|
| DoCopilot | Flagship applied-AI / RAG platform | Retrieval engineering, backend systems, auth, async ingestion, cloud deployment, evaluation, observability | It should not turn into a feature-sprawl RAG lab |
| Argus | Agent-systems proof | Workflow orchestration, HITL, async jobs, agent evaluation, failure handling | It should not become a second flagship or an agent-count project |
| ContextCore | MCP and memory proof | Tool-calling discipline, memory lifecycle, data-store trade-offs, tool safety | It should not become another web product |
| LoRA / ML project | ML credibility layer | Training adaptation understanding, metrics, model-evaluation literacy | It should not be rebuilt from scratch unless a JD specifically demands it |

## DoCopilot: exact update plan

## What DoCopilot is becoming

DoCopilot is becoming the main flagship project: a document-intelligence application that is not only able to answer questions from documents, but also able to show credible engineering around identity, data ownership, ingestion reliability, cloud deployment, and evaluation quality.[cite:22]

The end-state pitch should become: “This is an authenticated, cloud-deployed document Q&A system with hybrid retrieval, reranking, async ingestion, tenant-safe access boundaries, and measurable evaluation.” That is far stronger than saying it is a PDF chatbot or just a deployed RAG app.[cite:22]

### What already exists

The supplied repository material already shows PDF and TXT ingestion, Qdrant hybrid retrieval using dense and sparse signals, RRF fusion, reranking, FastAPI plus Next.js, streaming cited answers, Docker, local persistence, basic guardrails, and a 40-question evaluation plus ablation tables.[cite:22]

That means the base project is already strong enough to discuss in interviews. The update work exists to close credibility gaps, not to rescue a weak project.[cite:22]

### Change 1: Add PostgreSQL as the metadata source of truth

**What to add**

- `users`
- `tenants`
- `tenant_memberships`
- `documents`
- `document_versions`
- `ingestion_jobs`
- `evaluation_runs`

**Why it matters**

Right now Qdrant is good for retrieval but it should not be the main owner of user identity, document ownership, version history, or ingestion job state. Those are relational and transactional concerns, which are a better fit for PostgreSQL than for a vector database.[cite:22]

**What this gives**

- Real DBMS and SQL talking points.
- A strong answer for “why Postgres plus Qdrant instead of only one storage system?”
- Better ability to explain ownership, versions, and operational state.
- A direct bridge from project work to placement questions on normalization, indexing, transactions, joins, and schema design.[file:1]

**What should be better after it**

Before this change, the project sounds like an app with retrieval. After this change, it sounds like a real backend system with a proper transactional core and a specialized retrieval layer.

### Change 2: Replace client-supplied tenant filtering with real authorization

**What to add**

- JWT authentication.
- Tenant membership validation.
- Role checks, for example member or admin.
- Middleware that derives tenant scope from verified token claims.
- Cross-tenant integration tests.

**Why it matters**

The current repository description says tenant scoping exists via metadata filters. That is useful, but if the client can send `tenant_id`, then the system is trusting user input for access boundaries. That is not the same as authorization.[cite:22]

**What this gives**

- A defensible security answer.
- A concrete example of the difference between filtering and authorization.
- A strong interview line: “tenant scope is derived from verified identity, not accepted from request input.”
- Better alignment with real SaaS backend patterns.[cite:22]

**What should be better after it**

Before: “The app uses tenant filters.”  
After: “The backend enforces tenant scope from verified JWT identity and tests that Tenant A cannot access Tenant B resources through any endpoint.”

That second version sounds significantly more mature.

### Change 3: Build reliable asynchronous ingestion

**What to add**

- A file checksum.
- Version-aware uploads.
- An `ingestion_jobs` state machine: `queued`, `running`, `succeeded`, `failed`.
- Retry count and failure reason.
- Basic worker-based indexing flow.
- Idempotency protection for duplicate uploads.
- Validation on type, size, and malformed documents.

**Why it matters**

Long-running ingestion should not block the web API request path. Even if the first version is simple, the architecture should show that upload and indexing are separate concerns. This is especially useful because it maps directly to document-processing backend design, which is relevant to both GenAI and backend interviews.[file:17]

**What this gives**

- Async/backend workflow credibility.
- Good answers on retries, failure states, idempotency, and job status.
- A strong link between this project and the document-processing work described in the internship section of the resume.[file:17]

**What should be better after it**

Before: upload and indexing feel like one request-time feature.  
After: the project can be described as a proper ingestion pipeline with durability and observable state.

### Change 4: Deploy it as a cloud-backed demo, not just a local app

**What to add**

- FastAPI backend on Azure Container Apps.
- Next.js frontend on Vercel.
- Blob Storage for sanitized public demo documents.
- Key Vault for secrets.
- Health and readiness endpoints.
- GitHub Actions for test, build, and deploy.

**Why it matters**

Cloud deployment is not being added just for resume aesthetics. The actual benefit is being able to talk about containerization, environment management, managed secrets, deployment rollback, live debugging, and cost-aware infrastructure choices. Azure for Students provides free student credit and Azure Container Apps has a consumption free grant, which makes this a realistic student path if spending is watched carefully.[cite:58][cite:59]

**What this gives**

- A real answer to “how would you deploy this?”
- Cloud exposure for AI/ML roles that expect systems thinking, not only prompting.
- A stronger demo for recruiters, founders, and off-campus applications.

**What should be better after it**

Before: “I built a RAG app locally and deployed a frontend.”  
After: “I deployed an authenticated containerized FastAPI system with cloud storage, CI/CD, and managed secrets while keeping local fallback reproducibility.”

### Change 5: Replace one-number evaluation with stage-aware evaluation

**What to add**

A curated evaluation set of about 60 to 70 cases, split across:

- direct factual retrieval
- multi-chunk synthesis
- ambiguous queries
- insufficient-information / refusal cases
- conflicting-source cases
- citation validation
- a small safety/adversarial subset

Metrics should include:

- Recall@k
- MRR
- citation precision
- groundedness
- answer completeness
- P50/P95 latency
- failure rate

**Why it matters**

The current 40-question LLM-as-judge result is useful, but it is still a narrow result. A better evaluation lets the project answer the real interview question: “How do you know whether retrieval is failing, the reranker is failing, or generation is failing?” The curriculum explicitly emphasizes separating retrieval, grounding, answer quality, and observability rather than collapsing everything into one score.[file:1][cite:22]

**What this gives**

- Better scientific honesty.
- Stronger answers about sample size, benchmarking, and limitations.
- Evidence that the project was evaluated as a system, not just judged with one prompt.[file:1]

**What should be better after it**

Before: “I got 89.2% correctness.”  
After: “The project has a versioned corpus-specific benchmark with separate retrieval, citation, and grounded-answer metrics, plus manually reviewed samples.”

That is far more defensible in a high-quality interview.

### Change 6: Add observability and operational metrics

**What to add**

- Traces for upload, parse, chunk, embed, retrieve, rerank, and generation.
- Structured logs.
- Latency percentiles.
- Approximate token and cost tracking.
- Queue/job visibility.

**Why it matters**

Evaluations explain aggregate quality. Observability explains what happened on one bad request. The curriculum explicitly distinguishes traces, logs, and metrics from evaluation, and that distinction is useful in interviews.[file:1]

**What this gives**

- Better debugging narratives.
- A response to “how would you investigate a slow or bad answer?”
- Signs of real product/backend maturity.[file:1]

**What should be better after it**

The project moves from “interesting AI demo” to “small but credible AI platform.”

### What should not be added to DoCopilot now

Do not spend these eight weeks on GraphRAG, HyDE, fine-tuned embeddings, multi-agent RAG, Kubernetes, or an oversized admin panel. None of those changes solve the current placement bottlenecks. The bottlenecks are credibility, cloud exposure, auth, evaluation, and being able to defend the system under pressure.[file:1]

### Final value of DoCopilot after updates

After the planned updates, DoCopilot should prove:

- hybrid retrieval and reranking literacy
- backend API design
- auth and access-boundary design
- document-processing reliability
- cloud deployment and CI/CD
- evaluation discipline
- observability and debugging maturity
- the ability to explain a real applied-AI system end to end

## Argus: exact update plan

## What Argus is becoming

Argus is becoming the clear proof that multi-agent work can be designed as a controlled workflow rather than a buzzword. The goal is not to make it bigger. The goal is to make it measurable and robust enough to defend in an interview.[file:17]

The end-state pitch should become: “This is an async research workflow with planning, tool use, critique, human review, state persistence, and evaluated failure handling.”

### What already exists

The repository material already shows a supervisor-based LangGraph workflow, specialist agents, HITL interrupt handling, a three-iteration cap, SSE logs, async jobs, SQLite/Postgres support, Redis/Celery in Docker, and LangSmith tracing.[file:17]

That means Argus already proves architectural differentiation. It does not need a conceptual rewrite.

### Change 1: Add a 50-task benchmark

**What to add**

Build a benchmark covering:

- simple factual research
- multi-source synthesis
- contradictory sources
- recent/stale information
- paper retrieval
- no-result queries
- tool timeout or tool failure
- citation verification

Measure:

- task completion
- tool success
- citation support
- source diversity
- number of turns
- latency
- approximate token/cost use

**Why it matters**

Agent systems are hard to judge from screenshots. A benchmark shows whether the workflow actually finishes tasks well, whether it cites responsibly, and whether the loop behavior is controlled rather than decorative.[file:1]

**What this gives**

- Agent evaluation credibility.
- A better answer to “how do you know the workflow is good?”
- Stronger differentiation from simple one-shot research assistants.

**What should be better after it**

Before: “The graph has multiple agents.”  
After: “The graph has measured task quality, tool reliability, and bounded workflow behavior.”

### Change 2: Test workflow failure paths

**What to add**

Test cases for:

- tool timeout
- empty search results
- malformed tool output
- invalid supervisor route
- duplicate job submission
- worker restart mid-run
- HITL expiry

**Why it matters**

The project already has good workflow design ideas, especially the dedicated HITL node and capped retries. Testing failure paths proves that the architecture is not only elegant on paper but resilient in actual execution.[file:17]

**What this gives**

- Better system-design answers.
- Strong talking points on state, recovery, and terminal states.
- A clear reason why LangGraph or workflow graphs were chosen in the first place.[file:17]

### Change 3: Add simple public-usage protection if needed

**What to add**

- API key for public access.
- Basic per-key rate limits or quotas.

**Why it matters**

If the demo is public, tool-provider credits can be abused. A minimal protection layer is enough. Full account systems are unnecessary here because DoCopilot is the project that should carry the deeper auth story.

**What this gives**

- Practical security awareness.
- Better cost-control explanation.

### What should not be added to Argus now

Do not add more agents, a second framework, Tree of Thoughts, GraphRAG, or a major frontend rebuild. Those changes consume time while adding little placement value compared with evaluation and failure handling.

### Final value of Argus after updates

After the planned updates, Argus should prove:

- multi-agent workflow design
- HITL as a real control boundary
- async job orchestration
- state persistence and resume behavior
- failure handling in agent systems
- agent benchmark thinking
- practical trade-offs in workflow-based AI systems

## ContextCore: exact update plan

## What ContextCore is becoming

ContextCore is becoming the proof point for MCP, tool orchestration, and memory lifecycle design. It should stay a CLI-first system and should not be turned into another web product.[file:17]

The end-state pitch should become: “This is a memory-aware MCP-enabled assistant with explicit storage roles, tool boundaries, and policies for writing, correcting, and deleting memory.”

### What already exists

The repository material already shows a LangGraph flow, MCP client/server integration, task and note tools, PostgreSQL checkpointing, Qdrant semantic memory, MongoDB profile storage, streaming output, and an evaluation harness.[file:17]

This is already enough to prove that MCP and memory were actually built, not merely mentioned.

### Change 1: Fix the evaluation-count inconsistency

**What to add**

- Verify the real number of evaluation cases.
- Use the same number in the repo, resume, GitHub, and portfolio.

**Why it matters**

The current mismatch between the resume and project material weakens trust for no good reason. Small inconsistencies are exactly the kind that interviewers notice quickly.[file:17]

**What this gives**

- Credibility.
- Cleaner project narration.

### Change 2: Write explicit memory lifecycle rules

**What to add**

Define and test:

- what gets remembered
- what should never be remembered
- when memory is written
- how explicit corrections override old memory
- how deletion or forgetting works
- how retention should be thought about

**Why it matters**

Memory sounds powerful until it becomes stale, wrong, or unsafe. A serious memory project needs a policy for write, update, conflict, and delete behavior. The curriculum explicitly distinguishes episodic, semantic, and procedural memory concepts, and this project is the natural place to convert that theory into system behavior.[file:1]

**What this gives**

- Better depth on memory-agent design.
- Strong answers to “what happens when stored memory is wrong?”
- A safer and more realistic system story.[file:1]

### Change 3: Add tool-safety tests

**What to add**

Test cases for:

- invalid task IDs
- malformed tool arguments
- duplicate creates
- timeouts
- tool calls outside the allowlist
- confirmation flow for destructive deletes

**Why it matters**

MCP projects are not impressive just because the model can call tools. They become impressive when the tool boundary is controlled and the failure cases are understood.

**What this gives**

- Better MCP/system-design answers.
- Demonstrated awareness of tool safety and destructive actions.

### Change 4: Be ready to justify the three-store design

**What to prepare**

You should be able to explain:

- PostgreSQL = transactional tasks and graph checkpoints
- Qdrant = long-term semantic retrieval
- MongoDB = flexible profile data

Then also state the trade-off honestly: for a smaller real product, PostgreSQL with JSONB and possibly pgvector could be enough at first.

**Why it matters**

This prevents the project from sounding like a stack-collection exercise.

**What this gives**

- Architectural judgment.
- Better answers on simplification and trade-offs.

### What should not be added to ContextCore now

Do not build a full frontend, cloud deployment, additional memory types, or more databases. That would duplicate work already covered better by DoCopilot.

### Final value of ContextCore after updates

After the planned updates, ContextCore should prove:

- MCP implementation experience
- memory-agent design awareness
- tool safety thinking
- semantic versus transactional storage trade-offs
- the ability to explain and constrain stateful assistants

## LoRA / ML project: exact update plan

This project does not need a major rebuild unless a specific job description demands stronger model-training work. Its main role is to prevent the portfolio from sounding RAG-only or orchestration-only.[file:17]

### What to do

- Clean the README.
- Make sure dataset, split strategy, evaluation metrics, and training objective are clearly stated.
- Be able to explain LoRA, PEFT, overfitting, validation, and why fine-tuning was or was not better than retrieval for a use case.[file:1]

### Why this matters

Many AI/ML interviews still ask classical ML, training, metrics, and adaptation questions even when the resume is LLM-heavy. This project is the proof that model adaptation is not only theoretical knowledge.[file:1]

### What this gives

- A backup project for ML-heavy interviews.
- A stronger answer to “do you only know RAG, or do you also understand model training and evaluation?”

## Placement impact of these updates

The project updates are useful because each one creates a stronger interview sentence.

| Project | Better sentence after updates |
|---|---|
| DoCopilot | “Built and deployed an authenticated document Q&A platform with hybrid retrieval, reranking, tenant-safe access control, async ingestion, and versioned evaluation.” |
| Argus | “Built an async multi-agent research workflow with HITL, bounded retries, streaming logs, and benchmarked task/failure behavior.” |
| ContextCore | “Built a stateful MCP assistant with short/long-term memory, tool safety controls, and explicit policies for correcting and deleting memory.” |
| LoRA / ML project | “Implemented and evaluated parameter-efficient fine-tuning with clear metric and generalization trade-offs.” |

These are placement-useful not because they sound fancy, but because they compress engineering depth into one line that invites good interview questions.

## What all of this is NOT

This is not a plan to keep polishing projects forever. It is a bounded hardening plan whose job is to make the portfolio harder to dismiss. Once the project credibility gaps are fixed, more feature work has sharply lower return than DSA, OA, CS, system design, and interview practice.[file:1][file:17]

## Is “small project hardening + DSA/OA/core CS/system design” enough?

Yes, provided the project work is targeted and the rest is serious.

The actual placement stack should be:

- DSA and timed OA coding as the biggest daily priority.
- Core CS: DBMS, SQL, OS, CN, OOP.
- AI/ML fundamentals: metrics, probability/statistics, transformers, LoRA, RAG, evaluation, prompt injection, tool calling, agents.[file:1]
- System design: both AI-specific and classic patterns such as rate limiter, URL shortener, queue/notification system, chat system, and document-processing pipeline.[file:1]
- Project deep-dive preparation: 2-minute, 10-minute, and failure-case answers.
- Off-campus applications and referral flow starting early, not after the portfolio feels “finished.”

That combination is enough. In fact, for your profile it is the correct combination. Building more projects would likely reduce placement performance because it steals time from the exact stages that reject candidates first.

## Suggested effort split

| Area | Approximate share of effort |
|---|---:|
| DSA and OA | 30% |
| Core CS and SQL | 15% |
| AI/ML theory and system design | 15% |
| Project hardening | 25% |
| Applications, mock interviews, communication | 15% |

This split reflects the reality that projects create differentiation, but coding and interview fundamentals decide whether you survive long enough for that differentiation to matter.

## Final operating rule

Every project task should end as one of the following:

- a tested feature
- a benchmark result
- a deployment artifact
- a README improvement
- a design diagram
- a measurable security/evaluation note

If a task ends only as “learned concept” or “watched video,” it should not count as project progress.

## Practical execution order

1. Fix all public metric and wording inconsistencies immediately.
2. Create the sanitized public corpus for DoCopilot demos and evaluation.
3. Upgrade DoCopilot first because it is the flagship.[cite:22]
4. Run the contained Argus benchmark and failure suite.[file:17]
5. Clean ContextCore metrics and memory/tool-safety rules.[file:17]
6. Keep LoRA/ML as an interview support project rather than a rebuild target.
7. Spend the majority of daily disciplined time on DSA, OA, CS, system design, and applications.

## Anti-gravity rules

These are the rules that stop the plan from collapsing under scope creep:

- Do not add a fourth AI project.
- Do not add advanced RAG features before auth, eval, deployment, and observability are defensible.
- Do not let Argus become a playground for more agents.
- Do not let ContextCore become another product frontend.
- Do not cut daily DSA for project polish.
- Do not wait for a “perfect portfolio” before applying.
- Do not claim production-grade, hallucination-free, or secure multi-tenancy without exact evidence.
- Do not use confidential internship or client data in public demos or repositories.

## Bottom line

The portfolio is already strong enough. The right move is not to reinvent it. The right move is to harden each project just enough that each one proves a distinct capability, then use the saved time to dominate the actual placement funnel: OA, DSA, SQL/CS, system design, project communication, and application velocity.[file:1][file:17][cite:22]
