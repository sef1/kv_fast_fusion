import aiohttp
import json
import random

from benchmark_dataset import SampleRequest
    
class N8nAgentDataset:
    def __init__(self, agent_prompts):
        # agent_prompts is a list of dicts: {"code_agent": "...", "summarization_agent": "...", "problem_agent": "..."}
        self.agent_prompts = agent_prompts

    def sample(self, num_requests, **kwargs):
        # mimic output structure of other datasets
        requests = []

        for i in range(num_requests):
            random_key = random.choice(list(self.agent_prompts.keys()))
            prompt_set = random.choice(self.agent_prompts[random_key])

            # We create a synthetic "prompt" – this is what benchmark_serving_.py expects.
            # requests.append({
            #     "prompt": json.dumps({
            #         "iteration": i + 1,
            #         "agents": {
            #             # "code_agent": {"prompt": prompt_set["code_agent"]},
            #             # "summarization_agent": {"prompt": prompt_set["summarization_agent"]},
            #             # "problem_agent": {"prompt": prompt_set["problem_agent"]}
            #             random_key: {"prompt": prompt_set}
            #         }
            #     })
            # })
            requests.append(SampleRequest(
                prompt=prompt_set,
                prompt_len=len(prompt_set),
                expected_output_len=len(prompt_set),
                multi_modal_data=None)
                
            )

        return requests


async def async_request_n8n(payload, host, port, timeout=60):
    """
    Fully asynchronous n8n webhook call using aiohttp.
    Matches the benchmark behavior used for vLLM / OpenAI APIs.
    """
    url = f"http://{host}:{port}/webhook/ai-stress-test"

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:

            async with session.post(url, json=payload) as resp:
                text = await resp.text()

                if resp.status != 200:
                    return {
                        "error": f"HTTP {resp.status}: {text}",
                        "payload": payload,
                    }

                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {
                        "error": f"Invalid JSON response: {text}",
                        "payload": payload,
                    }

    except Exception as e:
        return {"error": str(e), "payload": payload}

code_agent_prompts = [
    """
Design a fully distributed caching algorithm suitable for massively parallel, geographically distributed compute clusters that consist of nodes with heterogeneous compute characteristics, variable memory layout, and diverse network topologies. The algorithm should support both push-based and pull-based synchronization models and should be able to adaptively switch between them based on observed workload profiles such as read-write ratios, cache churn, temporal locality, and consistency requirements. Additionally, the caching algorithm must incorporate a multi-level coherence protocol that spans L1 local node cache, L2 regional cache, and L3 global distributed cache, ensuring eventual consistency while minimizing cross-region invalidation storms.

Your solution must include a detailed explanation of how the system performs dynamic key partitioning, including hashing strategies, shard migration heuristics, rebalancing triggers, and congestion avoidance techniques. Provide pseudocode or real code implementing core mechanisms such as distributed hash table updates, quorum-based read and write operations, consistency repair, and anti-entropy synchronization.

Include simulation logic using available tools (e.g., code execution, HTTP calls to simulated nodes, or internal memory structures). Show how the system handles node failures, slow nodes, network partitions, traffic spikes, and cache poisoning attempts. Finally, integrate iterative self-correction using multiple tool calls: first design the initial system; then test it under simulated workloads; then refine based on results; then rerun tests. Return outputs in the required JSON format with "action", "tool_name", "tool_args", "tool_result", "final_answer", and "continue" fields. Continue the iterative loop until the simulation indicates stable and optimized performance.
""", 

"""
Create a deeply fault-tolerant, multi-stage message queue system designed for workloads requiring strict delivery semantics, throughput guarantees, and minimal tail latency. The system must support at-least-once, at-most-once, and exactly-once semantics, allowing dynamic switching based on workload requirements. Architect the queue as a distributed log-based structure with segmented storage, append-only behavior, and offset-based cursor management for concurrent consumers.

Your design should cover leader–follower replication with intelligent follower selection based on network latency, storage throughput, and recent heartbeat stability. Implement mechanisms for failure detection using phi-accrual failure detectors or alternative probabilistic mechanisms. Ensure that consumers can handle out-of-order delivery, consumer group balancing, lag monitoring, and commit offset reconciliation.

Simulate batch writes, log segment compaction, crash-recovery scenarios, checkpoint restoration, and back-pressure handling. Provide code for queue writers, readers, replication loops, metadata management, and monitoring logic. Use available tools to perform iterative testing: first produce a baseline design, then generate multiple simulated workloads, measure throughput and latency, detect bottlenecks, and refine. Use the agent loop to repeatedly call tools until a stable optimal configuration is reached. Output results in strict JSON format with tool calls and loop continuation flags.
""",

"""
You are a senior distributed systems architect helping design an enterprise-grade distributed caching algorithm for large-scale microservice infrastructures.
You must follow a multi-step reasoning process and iteratively refine your proposal using available tools.

Your objectives include:

Architectural Requirements

The cache must support horizontal scalability across hundreds or thousands of nodes.

It must tolerate node churn, partial network partitions, and temporary service degradation.

It must include consistent hashing or an improved strategy to balance load across nodes.

Nodes must be able to join/leave the cluster without triggering major rebalancing storms.

Performance Requirements

Achieve extremely low latency (< 5 ms for GET operations).

Guarantee predictable performance even under heavy write contention.

Provide bounded tail latency for the 99.9th percentile.

Fault Tolerance

The system must recover from node failures without data loss (within agreed SLA).

You must propose a replication mechanism (sync/async quorum-based replication, hinted handoff, etc.).

The algorithm must still operate even during partial network partitions.

Eviction & Consistency

Include a pluggable eviction policy (LRU, LFU, ARC, ML-guided, workload-adaptive, etc.).

Provide options for tunable consistency levels (eventual, causal, strong-read-your-write, etc.).

Describe how stale reads are prevented or mitigated.

APIs & Data Model

Describe the GET/PUT/MULTI-GET semantics.

Make the data model easy to embed in both stateful and stateless microservices.

Operational Concerns

Telemetry: per-key metrics, shard heat-map, replication lag tracking.

Hot-key detection and mitigation proposals.

Self-healing strategies and congestion control.

At least one deep math/algorithm derivation
(e.g., MIGRATION COST ANALYSIS using ring distance, expected number of keys moved on node join/leave)

Output requirements

First produce a JSON with fields:
action, tool_name, tool_args, tool_result, continue, final_answer.

If you need code (e.g., to simulate consistent hashing), call tools iteratively.

Continue until a fully-finished, implementation-ready design is produced.
""",
"""
Design an advanced load-balancing algorithm capable of operating in massive microservice environments.
System scale is 10,000+ containerized services across multiple regions and heterogeneous hardware.

Include:

Hybrid Predictive Load Balancing

Combine reactive metrics (CPU, queue depth, tail latency) with predictive modeling.

Support forecasting of load spikes using moving averages, decay functions, or ML inference.

Congestion & Admission Control

Drop, shed, or defer requests dynamically.

Provide formal guarantees (stability analysis, Lyapunov or queueing-theory based).

State Synchronization

Compare push-based gossip vs pull-based scraping.

Propose a hybrid model with adaptive intervals.

Fairness Guarantees

Prevent starvation.

Dynamically adjust for nodes with heterogeneous capacity.

Data Structures

Show pseudocode and diagram representations.

Derive time/space complexity.

Stress Scenarios

Black Friday surges with 20× traffic.

Cascading failures and feedback loops.

At least one simulation using tools
(even a mock simulation with numbers is fine)

End with JSON per the agent loop spec.
"""
]

summary_agent_prompts = ["""
Summarize the following highly technical article, which covers a wide set of advanced optimization techniques used across algorithm engineering, distributed systems, machine learning, and large-scale simulation workflows. The article discusses gradient-based optimization, metaheuristics like simulated annealing and genetic algorithms, constraint relaxation, convex and non-convex optimization strategies, and multi-objective Pareto frontier evaluation. Additionally, it explores distributed optimization frameworks, asynchronous gradient updates, parameter server architectures, and federated optimization in heterogeneous environments.

Your summary should highlight key insights, trade-offs, implications for real-world systems, and conditions under which each optimization technique is most effective. Identify the main engineering takeaways, outlining practical guidance for algorithm designers, performance engineers, and system architects working on large-scale infrastructures. Focus on clarity, precision, and actionable insights. Produce an iterative improvement cycle, calling tools if needed, refining the summary based on analysis. Output strict JSON with "continue" controlling the loop.
                         """, 
                         """
Condense the following executive-level report, which contains an analysis of organizational efficiency metrics, operational KPIs, cross-department collaboration bottlenecks, customer satisfaction trends, and technology adoption patterns. The report examines detailed measurements of workflow throughput, employee utilization, communication delays, system downtime patterns, SLA breaches, revenue-to-cost ratios, team performance variability, anomaly spikes in demand forecasting, and long-term strategic alignment with corporate OKRs.

Your summary must distill the report into a set of clear, actionable insights for leadership. Highlight the key opportunities for improvement, critical risks, bottlenecks with high leverage, and recommendations for process optimization, automation, and structural reorganization. Apply iterative refinement in a loop using available tools. Return the final result in strict JSON format after completing the reasoning loop.
""",
"""
You are an expert AI research analyst. You will be given a long technical article and must summarize it in multiple refinement passes using available tools.
Produce summaries at four levels of granularity:

Full detailed summary

Medium summary

Executive summary (for leadership)

One-sentence ultra-compressed insight

You must:

Identify all key hypotheses, methods, experimental setups, datasets, and metrics.

Detect hidden assumptions or methodological flaws.

Extract mathematical formulations and restate them in clear language.

Compare to prior work and identify the novelty.

Trace causal relationships (“X causes Y because Z”).

Identify implicit implications for systems research, practical deployments, and future work.

Additionally:

Produce a concept graph describing all major entities and relationships.

Generate a taxonomy of contributions, grouping findings into themes.

Identify weaknesses, unanswered questions, and contradictory statements.

During refinement:

Use tools to extract keywords, compute statistics, or rewrite sections.

Return JSON controlling the loop:
{action, tool_name, tool_args, tool_result, final_answer, continue}.

End only when an optimal, publication-ready summary is achieved.
""",
"""
Your task is to read a long technical report about system optimization and produce a distilled set of insights.
Your iterative steps must:

Extract raw facts

Performance bottlenecks

Experimental configurations

Failure modes and stability characteristics

Group findings into categories

Resource efficiency

Latency optimizations

Scaling patterns

Reliability risks

Infer unstated but logically implied consequences

“If the buffer depth is 64, and peak QPS is 10k, then…”

Propose optimized configurations

Hardware-aware parameter tuning

Caching strategies

Request sharding

Pipeline parallelism adjustments

Generate two outputs

A highly actionable, practical recommendation guide for engineers

A theoretical interpretation for researchers

Iterate until optimal output

Use tools (text rewrite, code interpreter for calculations, HTTP request if needed)

Use JSON loop format until complete
"""
                         ]

problem_solving_agent_prompts = [
    """
Solve the following multi-layered operational problem: You must optimize the end-to-end workflow of a real-time analytics pipeline that ingests high-frequency sensor data from multiple geographic regions, performs on-the-fly transformations, applies anomaly detection algorithms, and emits actionable alerts to downstream systems.

The pipeline suffers from unpredictable latency spikes, inconsistent memory usage, uneven load distribution, and intermittent data loss during failover events. You must identify root causes through structured reasoning, propose concrete mitigations, simulate alternatives using available tools, and evaluate trade-offs of various designs. Your goal is to produce a step-by-step solution incorporating iterative testing within the agent loop. Continue until a stable and actionable resolution is produced, then return strict JSON.
""", 
""",
Optimize resource allocation in a large-scale supply chain network consisting of multiple suppliers, factories, warehouses, transport channels, and retail endpoints. The system must support fluctuating demand, seasonal spikes, variable lead times, unreliable transportation, geopolitical risks, and partial information. Identify bottlenecks, propose restructuring strategies, design demand prediction enhancements, model cost–benefit trade-offs, and simulate the consequences of alternative allocation policies.

Use iterative tool calls to analyze discrete segments of the chain, compute throughput estimates, run hypothetical stress scenarios, and refine the plan. Continue iterating through the agent loop until a high-quality solution is achieved. Produce output in strict JSON format.
""",
"""
You are a senior operations research specialist tasked with optimizing a global supply-chain network under uncertainty.
You must break the problem into multiple LLM+tool cycles.

System details:

Global Network

14 manufacturing plants

27 warehouses

93 retail distribution centers

5 transportation modes (air/rail/sea/road/pipeline)

Unknown Variables

Demand volatility

Supplier reliability

Transportation disruptions

Inventory drift

Currency fluctuations

Cost Model

Production cost, transportation cost, warehousing cost

Penalty cost for stockouts

Holding cost vs lost-sales tradeoff

Environmental impact score

Constraints

Max throughput limits per facility

Storage temperature constraints

Regulations per region

Lead-time targets and contractual SLAs

Required Outputs

Propose a multi-objective optimization strategy

Include formulas (linear programming, mixed-integer programming, queueing networks)

Simulate small examples using tools

Suggest robust optimization approach for uncertain inputs

Provide step-by-step explanation with a DAG of reasoning

Finish with an implementation plan

Loop Requirements

At each iteration return JSON controlling whether another reasoning step is required

If more calculations needed, call tools dynamically

End only when a globally optimal (or Pareto-optimal) solution is reached
""",
"""
Design a multi-layer anomaly detection strategy for real-time data streams operating at millions of events per second.

Steps required:

Define anomaly types

Point anomalies

Contextual anomalies

Collective anomalies

Temporal anomalies

Propose detectors

Rolling Z-score

Robust statistical estimators

Autoencoders

LSTM-based temporal detectors

State-space models

Streaming clustering (DBSCAN variant)

Multi-Stage Fusion

Confidence fusion

Majority voting

Dynamic weighting based on latency constraints

Data Pipeline

Windowing

Feature embedding

Ratio metrics

Drift detection

Tool Use Requirements

May call the code interpreter to generate sample time series

May use HTTP calls to fetch external benchmark data

May construct JSON structures and refine them iteratively

Loop Control

Return JSON with "continue": true/false

Continue until the final system design is formally specified, including computational complexity and SLA compliance
"""
]
