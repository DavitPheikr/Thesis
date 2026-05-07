# Chapter 3 Working Document

## Part A: Internal Drafting Notes

These notes are for drafting and supervision only. They should not be copied into the final thesis. Part B contains the cleaned thesis-style prose.

### Current Status

This document describes the current methodology for the PandaSet-to-RandLA-Net pipeline. The full server baseline run has not yet been inspected locally, so the chapter must not claim final experimental results. Final metrics, qualitative examples, and interpretation belong in Chapter 4 after the server run and any controlled follow-up experiments are reviewed.

Use this chapter to explain the method, not to narrate the development history. Smoke tests, sanity checks, and medium runs are evidence for pipeline validation or configuration calibration. They are not the final experimental evidence unless explicitly selected later as part of the evaluated experiment.

### External Methodology-Writing Guidance Checked

The revision follows general academic methods-section guidance from reputable writing resources:

- George Mason University Writing Center, [Scientific (IMRaD) Research Reports - Methods Section](https://writingcenter.gmu.edu/writing-resources/imrad/imrad-method-section): methods should describe how empirical work was conducted in enough detail to make the work transparent and replicable, including data, procedures, tools, and justification of choices.
- George Mason University Writing Center, [Scientific (IMRaD) Research Reports - Overview](https://writingcenter.gmu.edu/writing-resources/imrad/writing-an-imrad-report): methods answer what was done, while results report findings and discussion interprets them.
- Sacred Heart University Library, [Organizing Academic Research Papers: The Methodology](https://library.sacredheart.edu/c.php?g=29803&p=185928): methodology should explain how data was obtained/generated and analyzed, and why particular methods or procedures were chosen.

Practical consequence for this chapter: explain dataset preparation, preprocessing, adapter design, model configuration, training setup, and constraints clearly; do not interpret final performance here.

### Verified Technical Facts

Verified from repository files, logs, configs, and scripts:

- PandaSet sequence directories: 103.
- Semantically annotated sequences: 76.
- Lane-bearing semantically annotated sequences: 66.
- Frozen split: 58 training sequences, 9 validation sequences, 9 test sequences.
- Frame counts: 4640 training frames, 720 validation frames, 720 test frames.
- Validation lane-bearing sequences: 7 of 9.
- Test lane-bearing sequences: 9 of 9.
- Forward-facing LiDAR sensor: sensor 1.
- Raw label names verified through the PandaSet devkit include 1 Smoke, 2 Exhaust, 3 Spray or rain, 4 Reflection, 7 Road, 8 Lane Line Marking, 9 Stop Line Marking, and 10 Other Road Marking.
- Implemented label remap: 1-4 -> ignore 0, 7 -> road 1, 8 -> lane 2, all remaining non-ignored raw IDs -> other 3.
- Raw Stop Line Marking and Other Road Marking are mapped to other, not lane.
- Intensity column: `i`.
- Intensity preprocessing: clip to [0, 114], then standardize with training-split mean 22.481164932250977 and standard deviation 14.912428855895996.
- Dataset sample fields: `point`, `feat`, `label`; split metadata is available through `get_attr`.
- Open3D/RandLA-Net input channels: 4, because xyz coordinates and the one intensity feature are combined by the model pipeline.
- Baseline model config: RandLA-Net, 3 active classes, ignored label 0, 16384 sampled points, 3 layers, subsampling ratio [4, 4, 4], grid size 0.04.
- Baseline sampler: `SemSegRandomSampler`.
- Loss: Open3D-ML `SemSegLoss`, using weighted cross entropy after ignored label filtering.
- Training-split class counts: ignore 398,730; road 119,562,394; lane 2,098,182; other 176,504,630.
- Configured class-weight values for active classes: [119,562,394.0, 2,098,182.0, 176,504,630.0] in order road, lane, other.
- Verified effective runtime CE weights on server: road 2.3753318786621094, lane 36.98638153076172, other 1.6340690851211548.
- Baseline training settings selected so far: batch size 1, validation batch size 1, `num_workers: 0`, `pin_memory: false`, learning rate 0.001, CUDA.
- Metric artifacts produced by the custom training pipeline include `eval_history.csv`, per-epoch JSON metric snapshots, and per-epoch confusion matrices.

### Facts Still Requiring Verification

These should stay marked in Part B until the server run is completed and pulled or otherwise inspected:

- [VERIFY: final baseline run status]
- [VERIFY: final number of completed epochs]
- [VERIFY: final steps per epoch used by the full baseline run]
- [VERIFY: final checkpoint availability]
- [VERIFY: final artifact set available locally]
- [VERIFY: whether the baseline is retained as the final reference or superseded by a tuned configuration]
- [VERIFY: final Chapter 4 metrics and qualitative outputs]

### Section Objectives

3.1 Pipeline Overview: give the reader the full pipeline before details.

3.2 Dataset Preparation and Splitting: establish the data source, exclusions, split strategy, and why sequence-level splitting matters.

3.3 Label Remapping and Target Class Definition: define the actual learning task precisely. This is central and must remain consistent with the evaluation chapter.

3.4 Point Cloud Preprocessing and Feature Representation: explain how raw LiDAR frames become model input.

3.5 Dataset Adapter and Model Input Contract: explain the methodological importance of the PandaSet-to-Open3D-ML adapter.

3.6 RandLA-Net Baseline Configuration: document the concrete baseline configuration without turning this into a related-work section.

3.7 Training Setup and Class Weighting: describe training setup and class imbalance handling.

3.8 Pipeline Validation and Implementation Challenges: distinguish mechanical validation and configuration calibration from final results.

3.9 Baseline Experiment Definition: define what Chapter 4 will evaluate, with verification markers where final server facts are not yet available.

### Revision Warnings

- Do not write this chapter as "I tried this, then I fixed that." Use methodological organization.
- Do not claim state-of-the-art performance or novelty of RandLA-Net.
- Do not call all road paint "lane marking." The active lane class is specifically PandaSet "Lane Line Marking."
- Do not put final metric interpretation into Chapter 3.
- Do not hide implementation constraints, but describe them as engineering constraints that influenced the method.
- Add real citations in the LaTeX version for PandaSet, RandLA-Net, Open3D-ML, semantic segmentation metrics, and class imbalance or weighted cross entropy.

---

## Part B: Revised Chapter 3 Draft

### 3 Description of the Investigation

This chapter describes the technical investigation used to study LiDAR-only lane-marking segmentation in PandaSet. The investigation is formulated as a supervised point-cloud semantic segmentation problem. Raw PandaSet LiDAR frames and semantic annotations are transformed into a simplified three-class task, adapted to the Open3D-ML data interface, and used to configure a RandLA-Net baseline model.

The role of this chapter is methodological. It explains how the data was prepared, how the target labels were defined, how the point-cloud samples were represented, how the model input contract was satisfied, and how the baseline training configuration was selected. The chapter also documents validation checks and implementation constraints that affected the baseline design. Quantitative performance results and their interpretation are reserved for Chapter 4.

### 3.1 Pipeline Overview

The investigation was designed as a reproducible LiDAR-only semantic segmentation pipeline. The input data consists of PandaSet LiDAR point clouds with point-wise semantic annotations. The output of the training pipeline is a segmentation model configured to predict one of three active classes for each evaluated point: road, lane marking, or other.

The pipeline begins by auditing PandaSet sequences for semantic segmentation availability and lane-marking presence. Semantically annotated sequences are then split into training, validation, and test subsets at sequence level. For each frame, the forward-facing LiDAR points are loaded together with their semantic labels. The raw PandaSet semantic IDs are remapped into the thesis-specific label space, the point coordinates are transformed into the ego-vehicle coordinate frame, and the LiDAR intensity value is clipped and standardized. The resulting dataset item provides point coordinates, an intensity feature, point-wise labels, and associated frame metadata.

The processed samples are provided to Open3D-ML through a custom dataset adapter. RandLA-Net is configured for the three active classes and trained with class-weighted cross entropy. During validation, the training pipeline records artifacts required for later analysis, including validation loss, per-class metrics, lane-specific metrics, and confusion matrices. These artifacts are used in Chapter 4 to evaluate whether the baseline pipeline identifies the rare lane-marking class or mainly learns the dominant scene classes.

The baseline is intentionally conservative. It is not presented as an optimal method or a new architecture. Its purpose is to establish a clear reference configuration for LiDAR-only lane-marking segmentation on PandaSet, so that later changes such as feature engineering, lane-aware sampling, or hyperparameter tuning can be evaluated against a stable starting point.

### 3.2 Dataset Preparation and Splitting

PandaSet was selected because it provides urban autonomous-driving scenes with LiDAR point clouds and semantic segmentation annotations [CITATION NEEDED: PandaSet]. Since the research question focuses on LiDAR-only lane-marking segmentation, the baseline uses LiDAR data and point-wise semantic labels only. Camera images are excluded from the model input. This restriction keeps the experiment focused on the information available in the point cloud and avoids mixing LiDAR-based effects with image-based visual cues.

The local PandaSet installation contains 103 sequence directories. A dataset audit identified 76 sequences with semantic segmentation annotations. Among these, 66 sequences contain at least one point labeled as lane line marking. This audit was necessary because lane markings are sparse and not all annotated sequences contain the target class. Without checking lane presence, validation or test subsets could provide weak evidence for the central thesis question.

The data was split at sequence level rather than by randomly mixing frames. The frozen split contains 58 training sequences, 9 validation sequences, and 9 test sequences. This corresponds to 4640 training frames, 720 validation frames, and 720 test frames. Sequence-level splitting is important because frames from the same driving sequence are temporally and spatially related. If adjacent frames from the same sequence appeared in both training and validation data, validation performance could be inflated by near-duplicate scene content. Keeping complete sequences within a single split gives a more defensible estimate of generalization to held-out driving sequences.

The validation and test splits were also checked for lane-bearing sequences. Seven of the nine validation sequences and all nine test sequences contain lane markings. This matters because the thesis target is not generic road-scene segmentation; it is the segmentation of a rare and specific class. Lane-specific metrics such as lane IoU, precision, recall, and F1 are only meaningful when the evaluation data contains sufficient lane support.

| Split      | Sequences | Frames | Role in the Investigation                              |
| ---------- | --------: | -----: | ------------------------------------------------------ |
| Training   |        58 |   4640 | Model optimization and training-statistics computation |
| Validation |         9 |    720 | Configuration monitoring and epoch-level evaluation    |
| Test       |         9 |    720 | Reserved final evaluation after model selection        |

### 3.3 Label Remapping and Target Class Definition

The model is not trained on the full PandaSet semantic taxonomy. Instead, the original labels are remapped into a thesis-specific semantic segmentation task with three active classes: road, lane marking, and other. A fourth label, ignore, is used for points excluded from loss computation and metric evaluation. This remapping defines the learning problem investigated by the thesis.

The lane class corresponds specifically to PandaSet raw label 8, "Lane Line Marking." It does not include all painted road markings. Stop-line markings and other road markings are assigned to the other class rather than the lane class. This distinction is methodologically important. The task is not generic road-paint detection; it is lane-line segmentation. Therefore, predicting stop lines or other road markings as lane should count as a lane false positive and should reduce lane precision.

The road class is mapped from PandaSet raw label 7, "Road." The other class contains all remaining non-ignored labels, including ground, vegetation, sidewalks, vehicles, pedestrians, static objects, stop-line markings, and other road markings. This class is intentionally heterogeneous. It represents the rest of the scene against which road and lane line markings must be distinguished.

The ignore class contains PandaSet labels 1 to 4: Smoke, Exhaust, Spray or rain, and Reflection. These labels are excluded because they describe atmospheric or sensor-related artifacts rather than stable scene surfaces or objects central to lane-marking segmentation. Excluding them prevents the loss and metrics from being influenced by classes that are not part of the research question.

| Raw PandaSet ID(s) | Raw Label Name(s)                                                                                                                | Thesis Label | Role         |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------- | -----------: | ------------ |
| 1                  | Smoke                                                                                                                            |            0 | Ignore       |
| 2                  | Exhaust                                                                                                                          |            0 | Ignore       |
| 3                  | Spray or rain                                                                                                                    |            0 | Ignore       |
| 4                  | Reflection                                                                                                                       |            0 | Ignore       |
| 7                  | Road                                                                                                                             |            1 | Road         |
| 8                  | Lane Line Marking                                                                                                                |            2 | Lane marking |
| 5, 6, 9-42         | Remaining non-ignored labels, including Ground, Stop Line Marking, Other Road Marking, vehicles, pedestrians, and static objects |            3 | Other        |

This reduction of the label space is a methodological choice. The research question concerns whether lane line markings can be segmented from the road surface and surrounding urban scene using LiDAR data. Keeping the full PandaSet taxonomy would introduce many classes that are not central to this question and would make the baseline harder to interpret. The three-class formulation preserves the critical distinction between road and lane line marking while requiring the model to reject other objects and non-lane road markings.

This label policy also makes the evaluation stricter in a useful way. Because stop-line markings and other road markings are mapped to other, a model cannot receive credit for treating all bright road paint as lane. This is important for interpreting lane precision in Chapter 4.

### 3.4 Point Cloud Preprocessing and Feature Representation

For each selected frame, the pipeline loads the forward-facing LiDAR points and the corresponding semantic annotations. The forward-facing LiDAR sensor is used because lane markings relevant to driving perception are primarily located in the road area ahead of the vehicle. Restricting the baseline to the forward-facing sensor reduces unnecessary scene coverage and keeps the input focused on the region most relevant to the task.

Point-label alignment is preserved during sensor selection and preprocessing. The semantic label for each point is retrieved using the same point indices that remain after selecting the forward-facing LiDAR data. This alignment is essential because semantic segmentation is a point-wise task: each coordinate, feature vector, and label must refer to the same physical point. If this alignment were broken, the loss function and evaluation metrics would no longer measure the intended segmentation problem.

The point coordinates are transformed into the ego-vehicle coordinate frame. Ego-frame coordinates provide a local spatial representation centered on the vehicle, which is more appropriate for learning scene structure around the driving platform than absolute world coordinates. The model is therefore encouraged to learn local spatial and reflectance patterns rather than sequence-specific global positions.

LiDAR intensity is retained as the baseline input feature. This choice is motivated by the possibility that painted lane markings may have reflectance patterns that differ from surrounding road surfaces. However, intensity is treated cautiously. Dataset analysis indicates that raw intensity may provide useful information for distinguishing lane line markings from road surface, but it is not sufficient by itself to distinguish lane lines from all other painted road markings. For this reason, intensity is included as one feature in the point-cloud representation rather than treated as a standalone lane detector.

Intensity preprocessing uses statistics computed from the training split. Raw intensity values are clipped to the interval [0, 114] and standardized using a mean of 22.481164932250977 and a standard deviation of 14.912428855895996. Clipping reduces the influence of extreme values, while standardization places the feature on a more stable scale for neural network training. Computing these values from the training split avoids using validation or test distribution information during preprocessing.

Each processed sample contains ego-frame coordinates, a standardized intensity feature, and remapped labels:

| Field   | Shape | Description                            |
| ------- | ----- | -------------------------------------- |
| `point` | N x 3 | Ego-frame xyz coordinates              |
| `feat`  | N x 1 | Standardized LiDAR intensity           |
| `label` | N     | Remapped semantic label for each point |

The Open3D-ML RandLA-Net implementation combines point coordinates and feature channels inside the model pipeline. Therefore, the baseline model is configured with four input channels: three coordinate channels and one intensity channel.

### 3.5 Dataset Adapter and Model Input Contract

A custom dataset adapter is required to connect PandaSet with the Open3D-ML semantic segmentation pipeline. PandaSet provides sequence-based access to LiDAR frames, poses, and semantic annotations. Open3D-ML expects dataset split objects that return point-cloud samples in a specific structure suitable for preprocessing, batching, model input, and metric computation. The adapter bridges these two representations.

For each requested frame, the adapter loads the LiDAR data and semantic annotation, selects the forward-facing sensor, applies the label remapping, transforms coordinates into the ego frame, standardizes intensity, and returns the processed arrays. The adapter also exposes the frozen training, validation, and test splits, so the same sequence-level partition is used consistently across training and evaluation.

The adapter is methodologically important because it preserves the alignment between coordinates, features, labels, and metadata. The point coordinates and intensity values must correspond to the same point indices as the semantic labels. In addition, metadata such as sequence ID and frame index is needed for later inspection, artifact naming, and qualitative analysis. A mismatch at this stage would invalidate both training and evaluation, regardless of the model architecture.

The adapter was designed to load frames on demand rather than eagerly loading the complete dataset into memory. This is important because the training split contains thousands of frames. On-demand loading reduces memory pressure and makes the pipeline more practical for local validation and server-side training.

| Adapter Output    | Purpose                                                          |
| ----------------- | ---------------------------------------------------------------- |
| Point coordinates | Provide spatial input to RandLA-Net                              |
| Intensity feature | Provide one LiDAR reflectance feature per point                  |
| Remapped labels   | Define the supervised target for each point                      |
| Metadata          | Preserve sequence and frame identity for validation and analysis |

### 3.6 RandLA-Net Baseline Configuration

RandLA-Net is used as the baseline architecture for point-cloud semantic segmentation. It is not presented as a newly proposed model or as the optimal architecture for lane-marking segmentation. Rather, it provides an established point-cloud segmentation backbone suitable for constructing a reproducible baseline within Open3D-ML [CITATION NEEDED: RandLA-Net] [CITATION NEEDED: Open3D-ML].

The baseline is configured for three active output classes: road, lane marking, and other. The ignore label is excluded from loss computation. The input representation consists of ego-frame xyz coordinates and the standardized intensity feature described above. Since the Open3D-ML implementation combines coordinates and features, the configured number of input channels is four.

The number of sampled points per model input is 16384. This value is used to provide sufficient point support for sparse lane markings while remaining feasible on the available GPU configuration. Smaller point counts are useful for mechanical testing, but they increase the risk that rare lane-marking points are underrepresented in sampled patches.

The baseline uses Open3D-ML's random semantic segmentation sampler. A spatially regular sampler was considered, but it was not selected for the baseline because it performs expensive preprocessing over all training and validation frames before the first epoch and does not directly address the rare lane-marking class. The random sampler gives a simpler and more practical reference configuration. Lane-aware sampling remains a possible later modification and should be evaluated separately if used.

| Component            | Baseline Configuration                 |
| -------------------- | -------------------------------------- |
| Framework            | Open3D-ML                              |
| Model                | RandLA-Net                             |
| Task                 | Point-wise semantic segmentation       |
| Active classes       | Road, lane marking, other              |
| Ignored label        | 0                                      |
| Input representation | Ego-frame xyz + standardized intensity |
| Input channels       | 4                                      |
| Sampled points       | 16384                                  |
| Number of layers     | 3                                      |
| Subsampling ratio    | [4, 4, 4]                              |
| Grid size            | 0.04                                   |
| Sampler              | Random semantic segmentation sampler   |

### 3.7 Training Setup and Class Weighting

The baseline is trained using the Open3D-ML semantic segmentation pipeline with weighted cross entropy. Class weighting is used because the target lane-marking class is rare compared with road and other. Without class weighting, the model could reduce the training loss primarily by learning the dominant classes while failing to learn useful behavior for the thesis target class.

Training statistics were computed from the training split. The measured class counts show a strong imbalance between lane markings and the two dominant active classes.

| Class        | Training-Split Count |
| ------------ | -------------------: |
| Ignore       |              398,730 |
| Road         |          119,562,394 |
| Lane marking |            2,098,182 |
| Other        |          176,504,630 |

The active-class counts for road, lane marking, and other are stored in the dataset configuration in that order. In the verified Open3D-ML runtime, these count-like values are transformed internally before constructing the cross-entropy loss. The resulting effective loss weights are approximately 2.3753 for road, 36.9864 for lane marking, and 1.6341 for other. This confirms that the lane class receives substantially higher loss weight than the dominant classes.

The selected baseline training configuration uses a batch size of 1, validation batch size of 1, no DataLoader worker subprocesses, and CUDA execution. Configuration calibration runs were used to select these settings. In particular, batch size 1 was retained because larger-batch calibration indicated a tendency toward excessive lane predictions, and DataLoader worker subprocesses were avoided because they were unstable in the available server environment.

| Setting                 | Baseline Value                                                      |
| ----------------------- | ------------------------------------------------------------------- |
| Loss                    | Weighted cross entropy through Open3D-ML semantic segmentation loss |
| Optimizer learning rate | 0.001                                                               |
| Batch size              | 1                                                                   |
| Validation batch size   | 1                                                                   |
| DataLoader workers      | 0                                                                   |
| Pin memory              | false                                                               |
| Device                  | CUDA                                                                |
| Random seed             | 42                                                                  |
| Training duration       | [VERIFY: final number of completed baseline epochs]                 |
| Steps per epoch         | [VERIFY: final full baseline train/validation steps]                |
| Checkpoint policy       | [VERIFY: final checkpoint frequency and available checkpoint files] |

### 3.8 Pipeline Validation and Implementation Challenges

Several validation checks were performed before treating the baseline as an experimental run. These checks were designed to verify that the pipeline was mechanically correct. They are separate from final performance evaluation.

The first validation stage checked dataset access. The PandaSet installation and patched devkit were tested to ensure that LiDAR frames, semantic annotations, sensor-specific point clouds, and intensity values could be loaded correctly. Representative frames were used to confirm that the forward-facing sensor could be selected and that the semantic labels remained aligned with the filtered point cloud.

The second validation stage checked label mapping and sample structure. The remapping function was applied to real frames and verified to produce labels only from the expected set {0, 1, 2, 3}. The dataset adapter was then checked across training, validation, and test splits to confirm the expected split lengths, point shapes, feature shapes, label shapes, and label ranges.

The third validation stage checked model compatibility. RandLA-Net was constructed using the processed sample format, and the input-channel configuration was verified against the Open3D-ML implementation. This was necessary because the model pipeline combines xyz coordinates and feature channels internally, making four input channels the correct configuration for xyz plus one intensity feature.

The fourth validation stage checked the training and validation path. Short local and server runs were used to confirm that dataset loading, preprocessing, forward propagation, loss computation, validation, checkpoint writing, and metric artifact generation were operational. These runs served as pipeline validation and configuration calibration rather than final experimental evidence.

Several implementation challenges influenced the methodology. The first was the PandaSet import path. The extracted dataset directory is named `pandaset`, which can shadow the importable PandaSet Python package. The project therefore uses the patched PandaSet devkit explicitly in the Python path. This is an environment constraint, but it affects reproducibility because using the wrong import path can break dataset access.

A second challenge was sampler behavior. The spatially regular sampler performs eager preprocessing over the full training and validation splits before the first epoch. For this dataset, that means processing thousands of frames before training begins. It is also class-blind and therefore does not directly address the rare lane-marking class. The random sampler was selected for the baseline because it starts training immediately and provides a simpler reference configuration.

A third challenge was runtime stability. Full-size training was moved to the server because the intended 16384-point RandLA-Net configuration is more suitable for GPU execution than local experimentation. In the server environment, DataLoader worker subprocesses were unstable, so the baseline uses `num_workers = 0`. This reduces input-pipeline parallelism, but it improves stability and reproducibility for the baseline run.

The final recurring challenge is class imbalance. Lane-marking points form a small fraction of the training data. The methodology addresses this initially through class-weighted loss and lane-specific validation metrics. It does not assume that class weighting fully solves the problem. If later experiments introduce lane-aware sampling or additional features, they should be reported as controlled modifications to the baseline.

### 3.9 Baseline Experiment Definition

The baseline experiment is defined as a three-class RandLA-Net semantic segmentation run on PandaSet forward-facing LiDAR. It uses the frozen sequence-level split, the road/lane/other label remapping, ego-frame xyz coordinates, standardized intensity, random sampling, and class-weighted cross entropy.

The purpose of the baseline is to provide a reference point for the thesis investigation. It establishes whether the current LiDAR-only Open3D-ML pipeline can produce meaningful lane-marking segmentation evidence under a conservative configuration. It is not assumed to be the best possible configuration. If later tuning, engineered features, or lane-aware sampling are introduced, they should be described as separate configurations and compared against this baseline.

The baseline training pipeline is expected to produce epoch-level validation artifacts, including validation loss, mean IoU, per-class IoU, per-class precision, recall, F1 scores, lane recall by distance bucket, and active-class confusion matrices. These artifacts are necessary because aggregate performance alone is insufficient for this thesis. A model may perform well on road and other while still failing to identify lane line markings.

| Aspect         | Baseline Definition                                           |
| -------------- | ------------------------------------------------------------- |
| Dataset        | PandaSet semantically annotated LiDAR sequences               |
| Sensor input   | Forward-facing LiDAR only                                     |
| Target classes | Road, lane marking, other                                     |
| Ignored labels | Smoke, Exhaust, Spray or rain, Reflection                     |
| Features       | Ego-frame xyz + standardized intensity                        |
| Model          | Open3D-ML RandLA-Net                                          |
| Sampler        | Random semantic segmentation sampler                          |
| Loss           | Class-weighted cross entropy                                  |
| Split          | Frozen sequence-level train/validation/test split             |
| Run status     | [VERIFY: final baseline run status and artifact availability] |

This chapter has defined the data preparation, label mapping, preprocessing, adapter design, model configuration, training setup, validation checks, and baseline experiment. Chapter 4 evaluates the quantitative and qualitative evidence produced by this baseline and by any later controlled variants that are retained in the final thesis.
