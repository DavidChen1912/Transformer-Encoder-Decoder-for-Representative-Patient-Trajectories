# OT-Transformer Encoder Decoder for CAR-T Cell-State Trajectories

## Why follow cell states over time?

CAR-T therapy modifies T cells to recognize and attack cancer cells. Longitudinal measurements can reveal sustained persistence, transient expansion, or limited expansion; persistent CAR T cells have been documented during long-term remission. Our question is which **state transitions** accompany different patient responses. [NCI overview](https://www.cancer.gov/about-cancer/treatment/types/immunotherapy/t-cell-transfer-therapy), [long-term persistence study](https://pubmed.ncbi.nlm.nih.gov/35110735/)

![Synthetic examples of sustained, transient, and limited CAR T-cell expansion over time. These curves illustrate the motivation and are not measurements from this cohort.](chart/car-t-dynamics.png)

## Project goal

### Identify how T-cell states evolve over time in each patient, and compare trajectories across patients.

We build patient-specific representative trajectories with a backbone and important branches through three stages:

- **scGHSOM clustering:** pool cells from all patients and time points to define shared cell states.
- **Optimal transport:** estimate population flow between adjacent visits and derive mass-weighted candidate paths.
- **Transformer Encoder Decoder:** summarize the candidate paths into each patient's representative backbone and important branches.

![Project workflow: scGHSOM clustering, followed by optimal transport, followed by Transformer Encoder Decoder.](chart/project-workflow.png)

# [More information: Transformer architecture and training](docs/transformer.md)

## 1. scGHSOM: establish shared cell states

![Cells from multiple patients and visits are pooled by scGHSOM into shared cell states. Schematic illustration.](chart/shared-cell-states.png)

## 2. Optimal transport: infer candidate trajectories

For each patient, solve OT between adjacent observed visits. The resulting population couplings provide transition masses; Markov chaining yields weighted candidate full paths. Many paths share segments, and their individual masses can become small as the number of visits grows.

![Adjacent state distributions produce an OT matrix; chaining adjacent couplings yields many candidate trajectories. Schematic illustration.](chart/ot-candidate-paths.png)

## 3. Transformer Encoder Decoder: learn a representative backbone and branches

**The challenge is to preserve important branching flow without showing thousands of overlapping candidate paths.** The model learns a compact representation of each patient's weighted path set, then selects a connected representative structure.

![Thousands of candidate paths are summarized by a Transformer Encoder Decoder into one backbone with an important branch. Conceptual output.](docs/chart/representative-trajectory.png)

### Transformer Encoder Decoder Architecture

```mermaid
flowchart TD
    I["Input: patient paths and shared state IDs"] --> EM["Embedding + positional encoding"]
    EM --> E1["Encoder block 1"]
    E1 --> E2["Encoder block 2"]
    E2 --> EO["Encoder output: contextual path tokens"]
    EO --> P["Processing: mean pooling + mass-aware aggregation"]
    P --> D["Decoder block 1"]
    EQ["Temporal-edge queries"] --> D
    D --> H1["Head 1: predict OT matrices"]
    D --> H2["Head 2: representative backbone and branches"]
    H1 -. "Stage 1: OT-loss backpropagation" .-> E1

    classDef encoder fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef processing fill:#fef3c7,stroke:#b45309,color:#451a03
    classDef decoder fill:#dcfce7,stroke:#15803d,color:#052e16
    class I,EM,E1,E2 encoder
    class EO,P,EQ processing
    class D,H1,H2 decoder
```

- **Input:** all candidate paths for one patient, state IDs, conditional transition probabilities, actual visit gaps, and full-path masses.
- **Encoder:** two four-head Transformer blocks encode each path. Mean pooling and mass-aware cross-attention form patient summaries.
- **Decoder:** temporal-edge queries read those summaries to produce patient-conditioned edge representations.
- **Two heads:** first reconstruct OT, then select representative edges with a mass-coverage and complexity objective.

| Training stage | Purpose | Parameters updated |
|---|---|---|
| OT reconstruction | Preserve the original transport distribution using KL loss | Shared encoder-decoder and OT head |
| Representative structure | Retain important path mass with fewer edges | Structure head; shared network frozen |

All patients train one shared model and receive individual graphs. The highest-original-mass retained path designates the backbone; connected retained edges supply branches. Coverage and excluded mass make the compression explicit. Patient-group labels are used afterward for comparison.

### Inside an encoder block

![Two-row pre-LayerNorm encoder block with attention, FFN, and two residual connections.](docs/chart/encoder-block.png)

**[For a detailed explanation of the Transformer architecture, mathematical formulation, loss functions, and training procedures, read the full manuscript (PDF).](docs/ot_transformer_manuscript.pdf)**

