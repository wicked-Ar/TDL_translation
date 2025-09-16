# TDL Translation Pipeline

This repository documents a reference architecture for translating natural-language task requests into robot-executable programs via an intermediate Task Description Language (TDL). The pipeline combines large language models (LLMs), planning algorithms, constraint analysis, and external solvers to safely deploy verified task code across heterogeneous robot platforms.

## System Overview

1. **Natural-language understanding** – A large language model (LLM), such as Google's Gemma, interprets human commands and extracts structured task goals, constraints, and success criteria.
2. **Intermediate representation** – The interpreted task is expressed in TDL, a platform-agnostic description that captures high-level semantics independent of robot vendors.
3. **Robot-specific compilation** – TDL programs are translated into vendor-specific job files that respect the physical and logical constraints of the target robot.
4. **Validation and deployment** – External solvers and simulators verify path feasibility and behavioral correctness prior to deploying validated code to real robots.

## State Machine

The end-to-end workflow is modeled as a state machine where each state encapsulates a stage in the planning and translation process. Feedback loops enable re-planning when validation fails.

```mermaid
stateDiagram-v2
    [*] --> State_init
    State_init --> "User Requirement Analysis"
    "User Requirement Analysis" --> "Path Planning"
    "Path Planning" --> "Motion Planning"
    "Motion Planning" --> "TDL Generation"
    "TDL Generation" --> "Robot Constraint Analysis"
    "Robot Constraint Analysis" --> "Target Job File Translation"
    "Target Job File Translation" --> "Validation & Verification"
    "Validation & Verification" --> "Robot Update"
    "Robot Update" --> [*]

    "Validation & Verification" --> "Path Planning" : Re-plan
    "Robot Constraint Analysis" --> "Motion Planning" : Re-plan
    "Target Job File Translation" --> "Robot Constraint Analysis" : Re-analysis
    "Target Job File Translation" --> "TDL Generation" : Re-translation
```

### State Definitions

| State | Purpose | Key Inputs | Key Outputs | Potential Transitions |
|-------|---------|------------|-------------|-----------------------|
| `State_init` | Idle state awaiting a new user request. | — | User requirement payload. | `User Requirement Analysis` |
| `User Requirement Analysis` | Parse user intent, extract tasks, success metrics, safety constraints, and environment context with the Gemma LLM. | Natural-language command, domain knowledge. | Structured task specification (`TaskSpec`). | `Path Planning` |
| `Path Planning` | Compute an ideal geometric/path-level plan that satisfies the specification. | `TaskSpec`, environment map, solver feedback. | Candidate path, feasibility metadata. | `Motion Planning`, `Path Planning` (loop on failure). |
| `Motion Planning` | Transform the geometric path into robot motion primitives (e.g., joint trajectories). | Planned path, robot kinematics. | Motion sequence (`MotionPlan`). | `TDL Generation`, `Path Planning` (on failure). |
| `TDL Generation` | Convert motion plan and task context into a TDL program using declarative constructs (tasks, actions, pre/post-conditions). | `MotionPlan`, `TaskSpec`. | `TDLProgram`. | `Robot Constraint Analysis`, `Motion Planning` (if representation fails). |
| `Robot Constraint Analysis` | Evaluate robot-specific limits (payload, reachability, safety envelopes). | `TDLProgram`, robot capability profile. | Constraint report, adjusted parameters. | `Target Job File Translation`, `TDL Generation` (if constraints violated). |
| `Target Job File Translation` | Compile TDL into vendor code (e.g., KUKA KRL, FANUC TP) via vendor adapters. | `TDLProgram`, robot profile. | `VendorCode`, translation diagnostics. | `Validation & Verification`, `Robot Constraint Analysis`, `TDL Generation`. |
| `Validation & Verification` | Run simulation and formal checks to ensure correctness and requirement alignment. | `VendorCode`, digital twin, external solver. | Verification report, approval token. | `Robot Update`, `Path Planning`, `Target Job File Translation`. |
| `Robot Update` | Deploy validated code to the physical robot and monitor execution. | Approved `VendorCode`. | Deployment log, feedback telemetry. | Terminal state (success) or `User Requirement Analysis` for next job. |

## Component Responsibilities

- **Gemma LLM Service**: Performs requirement analysis, semantic parsing, and generates preliminary TDL scaffolds. Provides explainability metadata for auditing.
- **Planning Engine**: Contains both path and motion planners, integrating with external solvers (e.g., OMPL, TrajOpt) and safety constraints.
- **TDL Compiler**: Maintains the TDL schema, validates syntax/semantics, and exposes translation hooks for downstream vendors.
- **Constraint Evaluator**: Stores capability profiles for each robot model and runs kinematic/dynamic checks.
- **Vendor Adapter Layer**: A plug-in architecture where each adapter handles file generation, syntax validation, and packaging for a specific robot family.
- **Verification Suite**: Leverages simulation, model checking, and regression tests to ensure compliance and correctness before deployment.
- **Deployment Orchestrator**: Handles staging, versioning, and rollback for robot updates, capturing telemetry for future learning.

## Data Artifacts

- `TaskSpec`: JSON document containing intent, constraints, environment references, and success conditions.
- `MotionPlan`: Structured sequence of robot poses, joint angles, and timing metadata.
- `TDLProgram`: Platform-neutral representation using TDL constructs (tasks, actions, conditionals).
- `VendorCode`: Robot-specific executable job file ready for upload.
- `VerificationReport`: Evidence collected from solvers/simulators confirming readiness for deployment.

## Feedback Loops & Re-planning

- **Constraint violations** detected during `Robot Constraint Analysis` trigger a return to `TDL Generation` or earlier planning stages with updated parameters.
- **Verification failures** route the workflow back to `Path Planning` or translation stages, preserving diagnostics for debugging.
- **Translation issues** (syntax, unsupported instructions) prompt re-analysis of constraints or adjustments in TDL.

## Implementation Notes

- The architecture assumes loose coupling via asynchronous messaging so that planners, translators, and verifiers can scale independently.
- For safety-critical deployments, integrate runtime monitoring and automatic halt mechanisms if telemetry deviates from expected execution traces.
- Incrementally expand vendor adapters; start with a reference implementation (e.g., ROS 2 control stack) before adding proprietary robots.

## Future Work

- Automate learning from execution feedback to improve LLM prompting and planner heuristics.
- Introduce formal specification templates enabling operators to define requirements with higher precision.
- Explore multilingual natural-language interfaces by fine-tuning the Gemma LLM on domain-specific corpora.

