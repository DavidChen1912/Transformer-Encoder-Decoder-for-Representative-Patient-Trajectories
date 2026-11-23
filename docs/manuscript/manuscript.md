# OT-Transformer Encoder-Decoder for Representative Patient Trajectories with Important Branches

@meta JING-WEI CHEN

### Abstract

Longitudinal single-cell measurements provide population snapshots rather than tracked cellular lineages. Optimal transport (OT) links shared cell states across adjacent visits, but composing the resulting couplings produces a combinatorial set of candidate trajectories. Ranking individual paths can obscure shared flow structure, while averaging divergent trajectories can remove meaningful branching. We formulate patient-specific trajectory summarization as learning a compact temporal graph from the entire OT-derived path distribution. An encoder-decoder Transformer first encodes state sequences and conditional transitions, aggregates paths into a mass-aware patient representation, and reconstructs adjacent joint OT matrices. A second, separately trained head selects temporal edges using a complete-path coverage objective regularized by edge complexity. Deterministic graph assembly designates one backbone and retains connected branches. For seven states and five visits, the formulation processes up to 16,807 paths and decodes 196 candidate edges. We specify the architecture, optimization, graph extraction, analytical properties, and evaluation protocol, and provide an exactly calculated illustrative example. The architecture and an initial model-training cycle are complete. An ongoing configuration study examines reconstruction fidelity and structural stability across hyperparameter combinations; this article presents the full method, analytical verification, and evaluation framework without asserting unreported quantitative performance.

**Keywords:** optimal transport; Transformer encoder-decoder; single-cell trajectories; branching graphs; patient representation; interpretable compression.

## 1. Introduction

A cell-state distribution can change even when total sampled cell abundance appears similar. In longitudinal CAR-T-associated T-cell measurements, our objective is to compare how populations occupy and move among shared states over time, and subsequently ask whether trajectory structures differ between response groups. Such comparisons require a common state vocabulary, a quantitative model of temporal redistribution, and a concise representation preserving important alternative routes. Response-group membership is not an input to our model; it is reserved for downstream descriptive comparison.

@image chart/car-t-dynamics.png

@caption Figure 1. Conceptual longitudinal abundance patterns. These synthetic curves motivate time-resolved analysis and are not measurements or response classifications from the patient cohort.

We use pooled scGHSOM clustering to define shared states, estimate adjacent-visit OT couplings for each patient, and compose conditional transitions into weighted candidate paths. With $K$ states and $T$ visits, exhaustive enumeration contains $K^T$ sequences. The intended setting $K=7$ and $T=5$ yields 16,807 paths. A small mass for a highly ranked complete path can reflect branching combinatorics rather than biological irrelevance. Conversely, a geometrical average of distinct trajectories need not correspond to a supported sequence of states.

Our objective is a compact temporal graph, rather than a ranked list of original paths or an interpolated curve. The graph has a designated backbone and can include supported splits and merges. A Transformer supplies a shared nonlinear representation of path context across the observed cohort. Its proposed advantage is an inductive bias toward common flow patterns under a constrained bottleneck, not access to biological evidence absent from OT. Direct OT optimization remains a necessary comparator.

The contributions are: ($i$) a fully specified OT-guided sequence-to-set-to-edge architecture; (ii) a two-stage objective separating transport fidelity from structural compression; (iii) explicit graph assembly and complete-path accounting; and (iv) analytical checks and a reproducible evaluation design. The model architecture and initial training are complete, and the configuration study evaluates reconstruction quality, stability, and biological usefulness under the specified protocol.

@image chart/project-workflow.png

@caption Figure 2. Project workflow: pooled scGHSOM clustering defines shared states, adjacent-visit OT estimates population redistribution, and the Transformer Encoder Decoder extracts representative patient structures.

## 2. Related work and positioning

Entropically regularized OT enables efficient coupling estimation through matrix scaling [1]. Waddington-OT uses transport to study temporal single-cell populations [2], motivating couplings as population-level hypotheses. Our upstream setting uses clustered marker profiles rather than directly adopting a gene-expression developmental model. We do not infer proliferation, death, or lineage ancestry merely from balanced cluster-level transport.

The Transformer contextualizes sequences through attention and positional information [3]. Learned-query set pooling is related to Set Transformer aggregation [4]. We combine short within-path self-attention with mass-weighted cross-attention over the patient path set. We do not apply quadratic self-attention across all 16,807 paths. Pre-LayerNorm residual blocks follow an established design [5]; optimization uses decoupled weight decay [6]. These are established components. The research question concerns their combination and evaluation for OT trajectory compression.

Growing hierarchical self-organizing maps supply the upstream hierarchy [7]. We use repository-specific scGHSOM outputs as fixed inputs, without claiming a new clustering algorithm. The nearest non-neural comparator is direct optimization of the identical coverage-complexity objective. That comparison isolates the contribution of learned representations from the objective itself. A stronger neural architecture is not assumed to produce a better explanation merely because it is more complex.

## 3. Problem formulation and transport-derived data

### 3.1 Observations, state vocabulary, and scope

Patient $p$ has $T_{p}$ visits at strictly increasing numerical days. Intervals may be irregular and differ across patients. Labels day1, day4 and day19 are valid; a fixed schedule is not imposed. Cell IDs, patient IDs and days are metadata, not clustering marker features. After pooling cells across patients and visits, clustering assigns each cell to one of $K$ shared states. Approximately $F=25$ measured marker features characterize each state through a center and feature weights. $K=7$ is the worked deployment setting, not a universal clustering constraint.

@image chart/shared-cell-states.png

@caption Figure 3. Shared-state construction from pooled patient cells collected at different times. The displayed cluster count is schematic; the application setting uses seven states.

$$a_{p,t,i}=\frac{n_{p,t,i}}{\sum_{j=1}^{K}n_{p,t,j}},\qquad a_{p,t}\in\Delta^{K-1}.$$

The $n$ values count sampled cells. We model relative population mass; multiplying a plan by the sampled source-cell total produces a count-scaled display, not a validated estimate of in-vivo cell numbers. Patients are the biological replicates. Thousands of paths from one patient are dependent derived objects and cannot be treated as thousands of independent patients.

### 3.2 State-based costs and adjacent OT

Let $c_i$ and $b_i$ be the marker center and feature-weight vectors exported for state $i$. Their elementwise product defines the weighted feature vector. The implementation uses its Euclidean distance for the feature term. Let $\ell_i$ denote the hierarchy layer of state $i$. The hierarchy penalty $H$ is zero for identical states, one for same-layer siblings, and $\ell_i+\ell_j-1$ otherwise, then divided by its maximum. This is a repository-specific penalty, not a general tree-geodesic distance. Missing hierarchy annotations must be audited rather than interpreted as biological evidence.

$$D_{ij}=\left\|c_i\odot b_i-c_j\odot b_j\right\|_2+\lambda_{\rm hier}H_{ij},\qquad C_{ij}=\frac{D_{ij}}{\max_{u,v}D_{uv}}.$$

The diagonal is set to zero; a zero normalizing maximum gives a zero cost matrix. Marker preprocessing, marker order, hierarchy labels and both clustering thresholds are fixed and recorded. The cost matrix is shared across patients in the same clustering run. The hierarchy coefficient and the later structure coefficient are distinct hyperparameters.

$$P_{p,t}=\underset{P\geq0}{\mathrm{arg\,min}}\ \langle C,P\rangle+\epsilon\sum_{i,j}P_{ij}(\log P_{ij}-1),$$

$$P\mathbf{1}=a_{p,t},\qquad P^{\mathsf{T}}\mathbf{1}=a_{p,t+1}.$$

Each $P$ is a joint distribution over source and target states and sums to one. Sinkhorn scaling uses the kernel $\exp(-C/\epsilon)$. Termination must be checked with marginal residuals, not only changes in a scaling vector. When finite-iteration plans violate compatibility, preparation balances an in-memory copy to observed marginals by alternating rows and columns scaling. Original OT files remain intact. The balanced copy, not the untouched file, supplies training targets and path probabilities. Record residuals before and after balancing, maximum entry adjustment, and iteration count. Numerical reconciliation cannot replace an adequately converged OT solver; impossible support must raise an error.

### 3.3 Conditional transitions and complete-path mass

$$Q_{p,t,ij}=\frac{P_{p,t,ij}}{a_{p,t,i}}\quad(a_{p,t,i}>0),\qquad Q_{p,t,i,:}=0\quad(a_{p,t,i}=0).$$

$P$ answers how much of the entire normalized population is assigned to an edge. $Q$ answers where mass goes conditional on its source state. Tokens use $Q$ to describe transition preferences, whereas reconstruction targets $P$ so source abundance remains part of the task. The zero-source convention prevents undefined division; compatible path distributions assign zero mass to unreachable states.

$$w_{p,r}=a_{p,0,s_0}\prod_{t=0}^{T_p-2}Q_{p,t,s_t s_{t+1}},\qquad r=(s_0,\ldots,s_{T_p-1}).$$

This first-order Markov composition gives a normalized path distribution reproducing each adjacent $P$ when marginals agree. Adjacent couplings do not uniquely identify a multi-time joint law; the formula selects one consistent law. Paths are inferred population trajectories, not observed cell lineages. Zero-mass paths may remain in human-readable exports but are removed before logarithmic mass weighting.

@image chart/ot-candidate-paths.png

@caption Figure 4. Adjacent state couplings, joint OT matrices, and candidate full trajectories. Illustrative three-state values are schematic; complete-path mass is obtained by Markov composition.

### 3.4 Input contract

@table
Object|Seven-state, five-visit shape|Role
State IDs|[16,807, 5]|Shared integer vocabulary; one row per path
Transition inputs|[16,807, 5, 3]|Conditional Q, scaled outgoing gap, valid
Path weights|[16,807]|Separate normalized complete-path masses
Joint OT targets|[4, 7, 7]|Balanced adjacent couplings
State descriptors|[7, approximately 25]|Marker centers for interpretation and ablation

Path exports include state sequences, adjacent joint-mass columns, conditional-probability columns and complete `path_mass`. Repeating a joint edge mass on several paths does not allocate that mass independently to every path. Shared marker descriptors are stored once per clustering run. Coordinates are visualization metadata; they do not define inferred temporal geometry in the primary network.

## 4. OT-Transformer Encoder-Decoder

The model has three levels of representation: contextual visit tokens within each path, a mass-aware summary of the patient path set, and patient-conditioned temporal-edge features. Figure 5 shows the complete computation and the two output heads. For seven states observed at five visits, the encoder processes up to 16,807 paths, while the decoder evaluates 196 candidate temporal edges. Shared weights couple learning across patients without combining their path sets. The reconstruction stage learns transport-preserving features; the structure stage uses those frozen features to select a compact graph.

@figure architecture

@caption Figure 5. OT-Transformer Encoder-Decoder Architecture, following the overview in docs/transformer.md. Blue denotes path encoding, amber patient-level processing and edge queries, and green decoding and outputs. The dashed feedback path represents Stage 1 updates; Stage 2 trains only the structure head.

### 4.1 Token construction and irregular sampling

For visit $t$ along a path, a trainable lookup $E(s_t)$ supplies 32 state dimensions. A separate MLP supplies 32 transition dimensions from the outgoing conditional probability, actual gap divided by a shared time scale, and a valid flag. Set $c_{\mathrm{time}}=100$ days initially. The flag is one when an outgoing interval exists and zero at the final visit. Both $Q$ and gap are zero at that last visit, distinguishing termination from a genuine zero-probability edge.

@image docs/chart/token-embedding.png

@caption Figure 6. Token construction. State identity and transition context each contribute 32 features. Concatenation gives 64 features, followed by additive positional encoding.

$$v_{p,r,t}=[q_{p,r,t},\frac{\Delta d_{p,t}}{c_{\rm time}},1],\quad t<T_p-1;\qquad v_{p,r,T_p-1}=[0,0,0].$$

$$x_{p,r,t}=[E(s_t)\Vert f_{\rm tr}(v_{p,r,t})]+{\rm PE}(t),\qquad x_{p,r,t}\in\mathbb{R}^{64}.$$

The transition MLP is Linear(3,32), GELU, Linear(32,32). Concatenation allocates 32 coordinates to state identity and 32 to transition context without a second projection. Both branches learn jointly in Stage 1. Separate branches provide transparent input allocation, not guaranteed equal information content or biological importance. Fixed sinusoidal position encoding is added elementwise afterward. Its index captures visit order; gap captures elapsed time. Neither assumes seven-day sampling. Absolute day is not separately embedded.

$${\rm PE}(t,2j)=\sin(t/10000^{2j/d}),\quad {\rm PE}(t,2j+1)=\cos(t/10000^{2j/d}),\quad d=64.$$

Marker centers affect upstream OT costs but do not enter primary tokens directly. A marker-augmented ablation may replace or enrich the state lookup under matched parameter budgets. Thus we do not claim an ID-only embedding explicitly encodes marker biology. State IDs have no ordinal biological meaning; their learned vectors are indexed parameters shared across patients.

### 4.2 Within-path attention and encoder blocks

One path supplies a $T_{p}$-by-64 token matrix $X$. Four heads independently project $X$ into queries, keys and values of width 16. A head shares projection weights across visits; different heads have distinct weights. Scores compare each visit query with all visit keys. Softmax operates across information-source columns within each query row.

$$A_h={\rm softmax}_{\rm columns}\left(\frac{(XW_h^Q)(XW_h^K)^{\mathsf{T}}}{\sqrt{16}}\right),\qquad O_h=A_h(XW_h^V).$$

$${\rm MHA}(X)=[O_1\Vert O_2\Vert O_3\Vert O_4]W^O.$$

With five visits, each score matrix is [5,5] and each head output is [5,16]. Concatenation and output projection restore [5,64]. Attention is bidirectional because the task retrospectively summarizes a complete time course; there is no future-prediction claim. Attention weights are internal information-routing coefficients, not biological transition probabilities or causal importance.

@image docs/chart/encoder-block.png

@caption Figure 7. One pre-LayerNorm encoder block. The two rows show successive attention and FFN sublayers, each followed by dropout and a residual addition. The primary encoder stacks two blocks.

$$U=X+{\rm Drop}({\rm MHA}({\rm LN}_1(X))),$$

$$Y=U+{\rm Drop}({\rm FFN}({\rm LN}_2(U))),\quad {\rm FFN}(u)={\rm GELU}(uW_1+b_1)W_2+b_2.$$

Use two independently parameterized pre-LayerNorm blocks. LayerNorm normalizes the 64 features of each token, not across patients or paths. Each residual adds the pre-sublayer representation to its transformed update. The FFN expands 64 to 128 and returns to 64 with GELU. Dropout is 0.1, including the standard encoder implementation's internal attention/FFN dropout; evaluation disables dropout. Two blocks are a starting capacity choice, not an assertion that additional depth is unnecessary. One-, two- and four-block networks are explicit ablations.

### 4.3 From contextual tokens to path vectors

$$R_{p,r}=\frac{1}{T_p}\sum_{t=0}^{T_p-1}Y^{(2)}_{p,r,t},\qquad R_p\in\mathbb{R}^{N_p\times64}.$$

Mean pooling occurs within a path after contextual encoding. It does not average state coordinates across different paths. Position encoding and attention let each token depend on ordered context before averaging; nevertheless mean pooling does not guarantee complete retention of order information. An architectural ablation prepends a learned [PATH] token before the first block and selects its final output instead of pooling. It is not appended after the encoder and is not used in the primary model. Current patient-wise processing handles each sequence length directly; introducing padding requires explicit attention and pooling masks.

### 4.4 Mass-aware patient aggregation

Let $S$ contain $M$ learned summary-query vectors shared across patients. Initially $M=2$ and $d=64$. These slots are not predetermined branches, states or patient prototypes. Single-head cross-attention reads path vectors, adding log `path_mass` as a prior to attention logits. Zero-mass paths are removed.

$$B_p=\frac{(SW_Q)(R_pW_K)^{\mathsf{T}}}{\sqrt{d}},\qquad \alpha_{p,mr}=\frac{w_{p,r}\exp B_{p,mr}}{\sum_u w_{p,u}\exp B_{p,mu}},$$

$$Z_p=\alpha_p(R_pW_V)\in\mathbb{R}^{M\times d}.$$

For 16,807 positive paths the score matrix is [2,16,807] and the output [2,64]. Patients use the same $S$ and weights but generally produce different summaries. A patient never directly attends to another patient's paths. The mass prior prevents counting every enumerated low-mass alternative equally; learned scores can still emphasize contextually informative routes. Slots can collapse to similar summaries, so two slots do not guarantee two modes. Reconstruction and capacity experiments determine whether this bottleneck retains sufficient information.

### 4.5 Temporal-edge queries and decoder

Each interval and ordered source-target state pair receives a query. Concatenate source and target embeddings in fixed order, apply a linear map, and add interval position encoding and a gap-MLP output. The gap MLP is Linear(1,32), GELU, Linear(32,64). The state lookup is shared with the encoder. A query contains no target $P$, $Q$ or gate value.

$$e_{p,tij}=W_{\rm edge}[E(i)\Vert E(j)]+b_{\rm edge}+{\rm PE}(t)+f_{\rm gap}(\frac{\Delta d_{p,t}}{c_{\rm time}}).$$

There are $L_p=(T_p-1)K^2$ rows, or 196 rows for four intervals and seven states. Each identifies one temporal edge, distinct from the encoder's 16,807 complete paths. In the two-state, three-visit example both counts happen to be eight; that equality is coincidental.

$$F_p=E_p+{\rm Drop}({\rm CA}({\rm LN}_E(E_p),{\rm LN}_Z(Z_p))),$$

$$H_p=F_p+{\rm Drop}({\rm FFN}_{\rm dec}({\rm LN}_F(F_p)))\in\mathbb{R}^{L_p\times d}.$$

Single-head cross-attention takes queries from normalized edge inputs and keys/values from normalized summaries. Scores have size [196,2], the value product [196,64], and the output projection preserves size. The decoder FFN uses 64-128-64 with GELU, dropout and a residual. There is no autoregressive generation or additional edge self-attention. Each $H$ row is one patient's contextual representation of an identified edge. Latent columns have no assigned biological meaning. Both task heads branch from $H$ before separate linear projections.

## 5. Two-stage objectives and optimization

### 5.1 Stage 1: joint OT reconstruction

$$\ell_{p,tij}=h_{p,tij}^{\mathsf{T}}u_{\rm OT}+b_{\rm OT},\qquad \widehat{P}_{p,tij}=\frac{\exp\ell_{p,tij}}{\sum_{u,v}\exp\ell_{p,tuv}}.$$

Softmax operates over all $K$ squared pairs within each interval, not separately over source rows. Patient output is [4,7,7]. A row-wise softmax would predict conditional $Q$ and discard source proportions from the target. Whole-interval softmax ensures nonnegative unit mass but does not enforce the source or target marginals. Marginal errors must therefore be measured explicitly.

$$\mathcal{L}_{\rm OT}=\frac{1}{P}\sum_{p=1}^{P}\frac{1}{T_p-1}\sum_{t=0}^{T_p-2}\sum_{i,j}P_{p,tij}\log\frac{P_{p,tij}}{\widehat{P}_{p,tij}}.$$

Use $0\log(0/q)=0$ for zero targets and `log_softmax` for stable predicted log probabilities. Average intervals within patients and then patients, giving equal patient weight regardless of visit/path count. Do not multiply this objective again by `path_mass`. The interval-logit gradient is predicted mass minus target mass, up to averaging constants. Gradients update the state table, transition MLP, encoder blocks, aggregation queries/projections, edge-query construction, decoder and OT head. The structure head is inactive.

This task autoencodes complete OT-derived inputs. Targets are mathematically related to inputs, so reconstruction demonstrates compression fidelity rather than independent biological prediction. Copying supplied OT achieves zero reconstruction error without compression and is an informative reference. Masking and denoising need separate leakage-aware designs because $P$, $Q$ and `path_mass` carry overlapping information; they are not assumed model capabilities here.

### 5.2 Stage 2: path coverage and edge complexity

$$g_{p,e}=\sigma(h_{p,e}^{\mathsf{T}}u_g+b_g),\qquad c_{p,r}=\prod_{e\in r}g_{p,e}.$$

A sigmoid gate represents soft edge retention. Gates are not OT probabilities and need not sum to one. Their product couples decisions along a full path: retaining one edge does not preserve a path if another required edge is removed. Let the positive-edge set contain temporal edges with positive original target mass; extraction cannot add edges outside this support.

$$\mathcal{L}_{{\rm cov},p}=\sum_r w_{p,r}(1-c_{p,r}),\qquad \mathcal{L}_{{\rm cmp},p}=\frac{1}{|\mathcal{E}_p^+|}\sum_{e\in\mathcal{E}_p^+}g_{p,e},$$

$$\mathcal{L}_{\rm structure}=\frac{1}{P}\sum_p[\mathcal{L}_{{\rm cov},p}+\lambda_{\rm structure}\mathcal{L}_{{\rm cmp},p}].$$

@image docs/chart/structure-objective.png

@caption Figure 8. Structure objective: preserve complete-path mass while limiting retained-edge complexity.

@image docs/chart/coverage-loss.png

@caption Figure 9. Coverage loss. Each complete-path weight is multiplied by the probability of failing to retain all required edges under the soft-gate surrogate.

@image docs/chart/complexity-loss.png

@caption Figure 10. Edge-complexity loss. The candidate-edge set is restricted to positive-mass temporal edges in the implementation.

The coverage term penalizes omitted complete-path mass; complexity discourages retaining every edge. Shared edges cost once even when many paths traverse them. This coefficient is independent of the upstream hierarchy coefficient. The surrogate does not enforce a tree, a prescribed branch count or global connectivity. Small $\lambda$ can favor dense graphs and large $\lambda$ an empty graph. A coverage-complexity frontier is more informative than one selected setting.

Stage 2 freezes shared parameters and the OT head and keeps the shared network in evaluation mode. Only the linear-sigmoid structure head is optimized. Zero initial weights and bias give gates of 0.5. At five visits the initial product is 0.5 to the fourth power, or 0.0625; products over longer paths may weaken gradients. Frozen $H$ may be cached; the current implementation recomputes it without gradients. Joint fine-tuning is a later controlled ablation rather than the primary method. Separating stages prevents simultaneous task gradients from changing a previously checked OT representation.

### 5.3 Optimization and transition between stages

Initial settings are AdamW [6], learning rate 0.001, weight decay 0.0001, gradient norm clipping at 1.0, and 100 epochs per stage. Each epoch accumulates patient losses divided by cohort size before one optimizer step. Thus patients share parameters without merging their path sets. Save post-update, dropout-disabled metrics and the best checkpoint for the active objective. These are initial configurations, not tuned results; Section 8 defines sensitivity experiments.

Stage 1 must meet prespecified per-patient and worst-interval criteria before interpreting Stage 2. Candidate engineering criteria are maximum interval KL no greater than 0.01 nats, maximum total variation no greater than 0.05, and maximum absolute marginal error no greater than 0.02. These are evaluation tolerances, not clinical thresholds or achieved scores. Report failures. The training implementation requires a Stage 1 checkpoint; the evaluation procedure applies the reconstruction-quality gate before structural interpretation.

## 6. Gates to one representative trajectory graph

### 6.1 Support and complete-path pruning

Nodes are (visit,state), so identical state IDs at different visits are distinct nodes. Select an edge if its gate meets threshold tau and original $P$ is positive. The initial tau is 0.5. Forward reachability starts at positive-mass states at the first visit; backward reachability starts at supported final states. Retain only edges reachable from a valid start that can also reach a valid end. This removes incomplete fragments without inventing unsupported edges. If no complete route remains, return an invalid/empty graph and report failure, rather than fabricate a fallback.

### 6.2 Backbone and connected branches

$$r_p^*=\arg\max_{r\subseteq G_p}\left[\log a_{p,0,s_0}+\sum_{t=0}^{T_p-2}\log Q_{p,t,s_t s_{t+1}}\right].$$

Dynamic programming over visits finds the maximum-original-mass retained path. Break ties deterministically using stored vocabulary order and record the rule. Retain the weakly connected component containing this backbone. Weak connectivity ignores direction only for component selection; every retained edge still points forward in time. Exclude other complete components and include their lost mass in final accounting. Outputs can contain incoming merges and several initial states as well as outgoing branches. The result is one connected temporal graph with one designated backbone, not necessarily a rooted tree or unbranched sequence.

@image docs/chart/representative-trajectory.png

@caption Figure 11. Representative output concept: a large path distribution is compressed into one trajectory graph with a backbone and an important branch. Repeated state IDs at distinct visits represent different temporal nodes. The curves illustrate structure rather than learned continuous geometry.

Export source/target day and state, learned gate, original balanced OT mass, and backbone membership for every retained edge. Also export the backbone and complete-path coverage, excluded mass, retained-edge count and branching-node count. Original masses are annotations; selecting edges does not automatically renormalize them into a conservative transport process. Maximum-mass designation does not prove a unique biological backbone.

### 6.3 Final output fidelity

$${\rm Coverage}(G_p)=\sum_r w_{p,r}\mathbf{1}\{\forall e\in r,\ e\in G_p\},\qquad {\rm Excluded}(G_p)=1-{\rm Coverage}(G_p).$$

Compute coverage after thresholding, complete-path pruning and component selection. A soft surrogate can conceal a disconnected or low-coverage discrete graph. Report retained edges, branching nodes with more than one outgoing edge, merging nodes with more than one incoming edge, and mass removed at each cleaning stage. Importance here means retained path mass justifies structural cost; biological importance requires marker-based interpretation and independent evidence.

## 7. Exact verification on a small complete example

The following is an analytical calculation, not a trained-model result. Consider two states and visits at days 0, 4 and 19. Write the adjacent joint plans by rows: $P_{0}$ = [(0.30,0.30); (0.10,0.30)] and $P_{1}$ = [(0.10,0.30); (0.30,0.30)]. Their marginals are $a_{0}$ = (0.6,0.4), $a_{1}$ = (0.4,0.6), and $a_{2}$ = (0.4,0.6). $Q_{0}$ has rows (0.5,0.5) and (0.25,0.75); $Q_{1}$ has rows (0.25,0.75) and (0.5,0.5).

For path 1-1-2, full mass is 0.6 times 0.5 times 0.75 = 0.225. Its $Q$ inputs are 0.5 and 0.75, not joint masses 0.3 and 0.3. Scaled gaps are 0.04 and 0.15. The final transition tuple is (0,0,0). This distinction between joint, conditional and complete-path mass is essential for interpreting every exported tensor.

@table
Path|First Q|Second Q|Complete mass
1-1-1|0.50|0.25|0.075
1-1-2|0.50|0.75|0.225
1-2-1|0.50|0.50|0.150
1-2-2|0.50|0.50|0.150
2-1-1|0.25|0.25|0.025
2-1-2|0.25|0.75|0.075
2-2-1|0.75|0.50|0.150
2-2-2|0.75|0.50|0.150

Eight encoded paths produce [8,3,64], pooling produces [8,64], and aggregation produces [2,64]. Two intervals times two sources times two targets yield eight candidate edge queries. Decoder features are [8,64]; the OT head groups eight logits into two 2-by-2 joint plans, whereas the structure head produces eight gates. Tensor consistency alone does not establish the usefulness of the learned representation.

Index edges in interval/source/target order. A hypothetical binary selection (1,1,0,1,0,1,1,1) retains paths 1-1-2, 1-2-1, 1-2-2, 2-2-1 and 2-2-2. Their masses sum to 0.825, six of eight edges remain, and the backbone is 1-1-2. For $\lambda_{\mathrm{structure}}=0.8$ the objective is $0.175+0.8\times0.75=0.775$. Keeping all edges gives 0.8; keeping none gives 1.0. This verifies the tradeoff but does not establish global optimality or actual learned gates.

## 8. Experimental design and evaluation

### 8.1 Cohort audit and estimand

The prepared run `CART_0320-0.5-0.5` currently contains nine folders, `ROS_CART1903` through `ROS_CART1911`. Earlier eight-patient diagrams were pedagogical. Final inclusion criteria must document exclusions, exact marker panel, sample counts, response definitions and ethical permissions. No unverified clinical metadata is inferred. The estimand is a descriptive representation of the observed cohort, not predictive performance on future patients. All included patients may train a shared model and subsequently receive individual outputs; reconstruction is explicitly in-cohort.

Patients are biological replicates. Seed variation estimates optimization stability, not clinical population uncertainty. Resampling cells within patient and visit estimates sampling sensitivity conditional on that patient. If clustering is recomputed, align new states before comparing graphs; fixing vocabulary instead gives conditional uncertainty and must be described accordingly. Group comparisons require supplied labels, patient-level sample sizes and effect sizes, and remain exploratory with a small cohort.

### 8.2 Reconstruction fidelity and the stage gate

$${\rm TV}(P,\widehat{P})=\frac{1}{2}\sum_{i,j}|P_{ij}-\widehat{P}_{ij}|.$$

$$\delta_{\rm marg}=\max\{\|\widehat{P}\mathbf{1}-a_t\|_\infty,\|\widehat{P}^{\mathsf{T}}\mathbf{1}-a_{t+1}\|_\infty\}.$$

Report KL, TV and marginal residual for every patient/interval, with patient means and worst intervals. An average cannot hide a poorly reconstructed patient. Compare bottleneck capacities before attributing failure to the structure head. Reconstruction alone does not validate a graph, and accurate in-cohort output can reflect memorization without supporting generalization claims. Apply the stage gate to the saved evaluation checkpoint before any structural interpretation.

### 8.3 Required baseline comparisons

@table
Comparator|Purpose and matching rule
Mass-ranked path union|Union highest-mass complete paths; compare at matched final edge counts after identical cleaning.
Edge-mass filtering|Threshold original joint edge mass; evaluate complete-path coverage rather than retained edge mass alone.
Direct gate optimization|Optimize the identical coverage-complexity loss with independent patient edge logits; isolate neural parameter sharing.
Discrete optimization|Optimize coverage subject to an edge budget; disclose solver time limits and optimality gaps.
Simple learned compression|Replace path attention with an MLP or mass-weighted mean; match decoder width and budget.

For discrete benchmarking, binary $x_{e}$ selects edges and binary $y_{r}$ indicates path coverage. Maximize the sum of $w_{r}$ $y_{r}$ subject to $y_{r}$ no greater than $x_{e}$ for every edge of path $r$ and a fixed edge budget. At a maximum, $y_{r}$ can be one exactly when every required edge is present. If the solver does not enforce connectivity, apply identical postprocessing and distinguish its pre-cleaning optimum from final graph scores. Enumerate all edge subsets for small teaching problems. Never call a time-limited full-problem incumbent an exact optimum.

### 8.4 Structural metrics and uncertainty

Primary comparisons are final complete-path coverage versus retained edges, and edge/branch stability across seeds and upstream resampling. Compare edge Jaccard similarity, backbone-edge agreement, branch inclusion frequencies and coverage distributions in a common aligned vocabulary. Show mass excluded by component selection. Higher coverage obtained through a larger graph is not superiority at matched complexity. Report the frontier instead of selecting a favorable isolated setting.

Predefine five optimization seeds, initially 11, 23, 42, 71 and 101. For selected configurations, begin with 100 within-patient/visit cell-resampling replicates conditional on the shared vocabulary, subject to a recorded compute budget. Do not treat paths, edges, seeds or bootstrap replicates as independent patients. Group-label permutation tests, if justified, must operate at the patient level and state exchangeability assumptions. Small-cohort group separation is descriptive evidence rather than clinical validation.

### 8.5 Hyperparameters and ablations

Use a staged search rather than a full Cartesian product: establish reconstruction feasibility across width/$M$, compare structural frontiers with a feasible trunk, then test marker augmentation, mean pooling versus a [PATH] token, mass-aware versus unweighted aggregation, and $Q$/gap ablations. Do not tune by maximizing response-group separation. Record failures and empty graphs. The target is a defensible fidelity-complexity tradeoff, not an arbitrarily low training loss.

@table
Parameter|Initial setting|Sensitivity study
Width / heads|64 / 4|Width 32,64,128 with divisible head widths
Encoder blocks / FFN|2 / 128|1,2,4 blocks; FFN width 2d
Summary slots|2|2,4,8,16; inspect bottleneck fidelity
Decoder heads|1|1 versus 4 at matched width
Dropout / decay|0.1 / 0.0001|Dropout 0,0.1,0.2; decay 0,0.0001,0.001
Learning rate / epochs|0.001 / 100 per stage|0.0003,0.001; fixed recorded budgets
Time scale / path chunk|100 days / 256|Scale 10,100; memory/time profiling
Structure lambda / tau|0.8 / 0.5|Lambda 0.1,0.3,0.5,0.8,1.0; tau 0.3,0.5,0.7
Upstream settings|$\tau_1=\tau_2=0.5$; $\epsilon=0.05$|Separate clustering, hierarchy and entropy sensitivity

## 9. Implementation and model development

The encoder-decoder architecture, two-stage objectives, graph-extraction procedure, and an initial model-training cycle have been completed. Current model development compares parameter combinations using the reconstruction, coverage-complexity, and stability measures in Section 8. The initial configuration is a reproducible reference rather than a declared optimal setting. Quantitative comparison tables and selected-model outputs will accompany the configuration-study release.

The numerical example in Section 7 verifies probability and objective calculations independently of model fitting. It is an analytical example, not a patient-level performance result. Preparation metadata records a largest pre-balancing marginal residual of approximately 0.0004764 in the current data run. The numerical provenance record includes the corresponding post-balancing residuals and entry adjustments. Balanced training targets remain distinguishable from the original solver exports.

The configuration-study reporting framework comprises per-patient reconstruction and stage-gate outcomes; coverage-complexity and stability comparisons against direct OT baselines; and patient-group descriptions supported by marker interpretation. Learning curves and annotated patient graphs include original OT mass, selected gates, and excluded-mass accounting. Model selection follows these quantitative criteria rather than visual appearance or maximization of response-group separation.

## 10. Discussion and limitations

The central hypothesis is that shared constrained representations can favor reproducible motifs across noisy patient-specific OT distributions. Temporal attention captures ordered path context, the mass-aware bottleneck summarizes a distribution without a single coordinate average, and edge decoding ties predictions to that summary. This could regularize structure selection across the observed cohort. It may instead underfit rare patient-specific routes, collapse summary slots, or learn an expensive approximation to direct optimization. These competing possibilities motivate the matched-objective baseline.

The model introduces no independent lineage information. Balanced OT depends on the chosen costs and normalized marginals. Markov chaining chooses a multi-time law rather than observing one. Entropic regularization can make support diffuse, so numerical positivity does not establish biological plausibility. Gates do not calibrate biological certainty. Repeated state occupancy cannot be interpreted as tracking the same physical cell. Missing visits create longer inferred intervals, not measured intermediate behavior.

The objective forces neither a unique trunk nor a tree. Hard thresholding can differ sharply from soft coverage; restricting output to one weak component may exclude a disconnected biological subpopulation. We report the lost mass rather than imply full representation. Learned summaries are invariant to path enumeration order but require aligned state vocabularies. Approximately 25 markers support interpretation; mechanistic annotation still needs verified marker definitions and domain expertise.

A small observed cohort does not establish a foundation model, treatment efficacy, clinical prediction or broad generalization. Shared parameters are a modeling choice, not proof of transferable biology. Clinical labels, data provenance, consent, review-board approval and access restrictions are not supplied in the notebook and must be verified before clinical claims or release. Public figures should use pseudonymous identifiers. This is exploratory population analysis, not a clinical decision rule.

## 11. Conclusion

We present an implemented Transformer encoder-decoder compressing OT path distributions into patient-specific temporal graphs with a designated backbone and retained branches. The architecture separates transport reconstruction from structure selection and measures complete-path coverage after graph assembly. Exact calculations verify the mass definitions and selection objective. Following initial training, the configuration study assesses fidelity, sparsity, stability, and biological interpretability against direct OT summarization. The final configuration is selected through the stated reconstruction and structural evaluation criteria.

## References

[1] M. Cuturi. Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances. Advances in Neural Information Processing Systems 26, pp. 2292-2300, 2013. https://arxiv.org/abs/1306.0895

[2] G. Schiebinger et al. Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming. Cell 176(4), pp. 928-943.e22, 2019. doi:10.1016/j.cell.2019.01.006. https://pubmed.ncbi.nlm.nih.gov/30712874/

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention Is All You Need. Advances in Neural Information Processing Systems 30, 2017. https://arxiv.org/abs/1706.03762

[4] J. Lee, Y. Lee, J. Kim, A. Kosiorek, S. Choi, and Y. W. Teh. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. ICML, PMLR 97, pp. 3744-3753, 2019. https://proceedings.mlr.press/v97/lee19d.html

[5] R. Xiong et al. On Layer Normalization in the Transformer Architecture. ICML, PMLR 119, pp. 10524-10533, 2020. https://proceedings.mlr.press/v119/xiong20b.html

[6] I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. ICLR, 2019. https://arxiv.org/abs/1711.05101

[7] M. Dittenbach. The Growing Hierarchical Self-Organizing Map: Uncovering Hierarchical Structure in Data. Diploma thesis, Vienna University of Technology, 2000. https://www.ifs.tuwien.ac.at/~andi/download/thesis/dit_thesis00.pdf

## Appendix A. Probability and optimization properties

### A.1 Normalization and adjacent-plan recovery

Assume all joint plans have compatible marginals. Summing the path formula backward from the final state successively removes normalized $Q$ rows on positive-mass support, leaving the initial marginal, whose sum is one. At visit $t$ the induced state marginal equals the observed marginal by induction: multiplying the previous marginal by $Q$ gives the columns sum of the preceding $P$. Thus the adjacent marginal at ($t$,$i$,j) equals $a_{t,i}Q_{t,ij}=P_{t,ij}$. This proves consistency for our chosen Markov construction, not uniqueness among all compatible multi-time laws.

### A.2 Invariance to path enumeration

Permuting path rows permutes encoded vectors, keys, values and aligned masses together. Attention logits and weights receive the same columns permutation, which cancels in the product with value rows. Therefore patient summaries and downstream outputs are unchanged in deterministic evaluation, up to floating-point summation order. Masses and path identities must remain aligned. This does not imply invariance to reordering visits inside a path.

### A.3 Coverage derivatives and multi-affine structure

$$\frac{\partial\mathcal{L}_{{\rm structure},p}}{\partial g_e}=-\sum_{r:e\in r}w_r\prod_{f\in r,f\ne e}g_f+\frac{\lambda_{\rm structure}}{|\mathcal{E}_p^+|}.$$

An edge receives a stronger retention reward when it supports substantial complete-path mass through other retained edges. The gate-logit gradient additionally multiplies this expression by $g_e(1-g_e)$, so saturated gates may update slowly. The free-gate objective is multi-affine because a temporal path traverses an edge at most once: it is affine in each individual gate with the others fixed. A minimum on the closed gate cube has at least one binary vertex minimizer, since each coordinate can move to a minimizing endpoint without increasing the objective. This supplies neither uniqueness nor an efficient global optimizer. A sigmoid reaches endpoints only in the limit; shared frozen-feature heads restrict feasible gate patterns, so the free-gate result is not an attainability guarantee for neural parameters.

### A.4 Exact coverage without enumerating paths

$$\beta_{0,i}=a_{0,i},\qquad \beta_{t+1,j}=\sum_i\beta_{t,i}Q_{t,ij}g_{t,ij},\qquad C_{\rm soft}=\sum_j\beta_{T-1,j}.$$

Expanding this recursion gives exactly the sum of path mass times gate products. It evaluates coverage in $O(TK^2)$, rather than $O(K^T T)$. Replacing gates with binary selected-edge indicators gives exact hard coverage. This is an optional loss/evaluation acceleration and a strong direct-optimization baseline; the existing implementation uses explicit path products. It does not remove the present encoder's exhaustive path input. A model taking only adjacent plans is a distinct architectural experiment, not an unnoticed replacement of the path formulation.

### A.5 Exact small-instance test

For two states and three visits there are eight temporal edges and 256 binary edge subsets. Enumeration can verify the globally optimal free binary objective and compare it with a trained head's attainable graph. For the six-edge example in Section 7, coverage is 0.825 and objective 0.775 at $\lambda$ = 0.8. These values can be checked by direct path summation and by the forward recursion. Agreement verifies accounting, not neural learning. For full 196-edge graphs, exhaustive enumeration is infeasible and solver optimality must not be assumed.

## Appendix B. Algorithms and reproducibility

### B.1 Preparation algorithm

Input is a cell-marker matrix with cell IDs, patient IDs and numeric day labels, plus clustering and OT settings. Output comprises shared descriptors, patient-specific ordered visits, balanced OT targets, state paths, conditional transitions, complete masses and numerical provenance. (1) Validate markers and metadata, rejecting unparseable days rather than enforcing a calendar. (2) Pool cells for one shared clustering run; record normalization, seed and thresholds. (3) Derive and align state centers, weights and hierarchy. (4) Estimate all patient-adjacent couplings with a common cost. (5) Audit and balance copies to compatible marginals. (6) Enumerate sequences, compute conditional and complete masses, and verify normalization and adjacent-plan recovery. (7) Save manifests and hashes; do not overwrite original raw OT files.

### B.2 Stage 1 algorithm

Initialize shared parameters and the OT head. At each epoch zero gradients, then for every patient encode the complete positive-mass path set, aggregate with aligned weights, construct every temporal-edge query, decode features and interval-normalized plans, and backpropagate that patient loss divided by cohort size. Clip accumulated gradients and perform one AdamW step. Evaluate updated parameters with dropout disabled and save the best reconstruction checkpoint and logs. Apply the predefined gate to that checkpoint. An epoch is one full-cohort accumulated update, not one update per path.

### B.3 Stage 2 algorithm

Restore the accepted checkpoint with exact vocabulary and configuration. Freeze the trunk and OT head, reset the structure head, disable trunk dropout and cache or recompute $H$ without gradients. Each epoch accumulates cohort-average coverage-complexity loss and updates only structure parameters. Select the best soft-objective checkpoint without implying it minimizes the final hard-graph objective. For each patient compute gates, apply support/threshold, remove incomplete fragments, designate a backbone, keep its weak component and measure hard coverage. Save failures as invalid and include them in comparisons.

### B.4 Computational cost and memory

Let $N=K^T$, $d$ denote width, $M$ summary slots, $B$ encoder depth, and $L=(T-1)K^2$. Path attention costs $O(NB(T^2d+Td^2))$; summary attention costs $O(MNd)$ after projections; decoder attention costs $O(LMd)$ plus projections and FFNs. Include width-dependent matrix projections in runtime accounting. There is no $N$-by-$N$ attention. A full [16,807,5,64] float32 token tensor alone occupies approximately 20.5 MiB, excluding gradients, $Q$/$K$/$V$, FFN activations and optimizer states.

The implementation chunks encoding in groups of 256 paths. This reduces operator sizes but does not guarantee bounded total training activation memory: autograd retains all chunk graphs until backpropagation of the patient loss. Profile named hardware, peak memory and wall time. Activation checkpointing or recomputed streaming aggregation would be additional engineering. Frozen Stage 2 permits caching [196,64] features per patient, approximately 49 KiB in float32, excluding metadata. Increasing $M$ expands the patient bottleneck without multiplying path length, while increasing depth increases per-path compute.

### B.5 Repository mapping

@table
Module|Responsibility
`prepare_data.py`|Balanced copies, `states.csv`, per-patient `paths.csv` and metadata
`dataset.py`|Aligned tensors, validation, positive-path filtering and device transfer
`model.py`|Embedding, path encoder, mass-aware aggregation, edge decoder, heads and losses
`train.py`|Two-stage freezing, gradient accumulation, checkpoints and logs
`extract_trajectory.py`|Graph cleaning, backbone, edge exports and hard coverage
`evaluate.py`|Reconstruction and extracted-graph evaluation

These modules are under `transformer/code/`. Prepared files are under `transformer/data/CART_0320-0.5-0.5/`, with shared descriptors and one folder per patient. A release manifest should record commit, dependency versions, hardware, precision, seeds, input hashes, inclusion criteria, state/marker ordering and scaling, tau1/tau2, hierarchy coefficient, epsilon, balancing settings, dimensions, optimizer settings, acceptance criteria, threshold and tie rules. Separate observed source files, balanced derivatives, checkpoints and extracted graphs.

### B.6 Experimental reporting record

For each configuration report parameter count, reconstruction statistics by patient/interval, accepted or failed stage gate, training budget actually consumed, final coverage, edge count, branch/merge counts, invalid-graph rate, component-removal mass, seed dispersion, resampling stability, runtime and peak memory. For each baseline report identical graph cleaning and matched complexity. For group analysis document label provenance, patient counts, prespecified graph statistic, effect size and uncertainty. Do not select a favorable seed or exclude failed patients after seeing group differences.

## Appendix C. Configuration study and reproducibility release

The initial trained model defines the reference architecture for the configuration study. Model width, summary capacity, encoder depth, regularization, and structure-selection thresholds are varied under fixed patient inclusion and preprocessing. Each configuration is identified by its settings, random seed, checkpoint, data manifest, and evaluation record. Comparisons retain the same patient weighting and graph-cleaning rules.

The selected-model release combines parameter settings, learning curves, reconstruction metrics, structural frontiers, stability summaries, and annotated patient outputs. The release distinguishes analytical demonstrations, numerical preparation diagnostics, and fitted-model measurements. Parameter selection is based on fidelity and structural stability; patient-group labels are reserved for the subsequent descriptive comparison.
