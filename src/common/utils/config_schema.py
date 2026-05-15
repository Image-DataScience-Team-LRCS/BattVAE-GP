from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Optional, List, Any
from pathlib import Path


class General(BaseModel):
    experiment_name: str
    training: bool
    inference: bool
    inference_interpolation: bool = False
    resume_training: bool
    debug_mode: bool
    show_progress: bool = True
    seed: int
    interpolation_dataset: Optional[str] = None
    training_datasets: Optional[List[str]] = None
    validation_datasets: Optional[List[str]] = None

class LRSchedule(BaseModel):
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    decay_factor: float
    patience: int
    min_lr: float


class EarlyStoppingConfig(BaseModel):
    patience: int
    min_delta: float
    warmup_epochs: int
    monitor_metric: str


class KLAnnealingConfig(BaseModel):
    warmup_epochs: int
    start_factor: float
    target_factor: float



class PhysicsHeadConfig(BaseModel):
    enabled: bool = True


class HyperParameters(BaseModel):
    input_seq_len: Optional[int] = None
    input_channel: int
    latent_dim: int

    train_split: float
    batch_size: int
    dropout_rate: float
    conditional_vector_dim: int

    lr_schedule: LRSchedule
    epochs: int
    early_stopping: EarlyStoppingConfig
    soh_factor: float
    use_mask_padding: bool
    padding_value: float
    save_every_reconstructions: bool
    kl_annealing: KLAnnealingConfig
    physics_head: PhysicsHeadConfig = PhysicsHeadConfig()
    d_model: int
    num_heads: int
    num_transformer_encoder_layers: int
    num_transformer_decoder_layers: int
    n_fourier_encoder: int
    n_fourier_baseline_decoder: int
    n_fourier_residual_decoder: int
    max_freq: float
    freeze_baseline_epoch: int



class Paths(BaseModel):
    visualization: str
    metrics: str
    pretraining_data: str
    predicted_data: str
    model_save: str
    latent_space_save: str
    gp_output_dir: Optional[str] = None
    gp_model_summary_csv: Optional[str] = None
    gp_interpolation_output_csv: Optional[str] = None
    vae_interpolation_dir: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def create_path_if_not_exists(cls: str, v: str) -> str:
        path = Path(v)
        target_dir = path.parent if path.suffix else path
        target_dir.mkdir(parents=True, exist_ok=True)
        return str(path)


class DatasetConfig(BaseModel):
    id: str
    path: str
    interpolation_latent_path: Optional[str] = None
    discharge_symbol: int
    cycle_min: int
    cycle_max: int
    temperature: float
    charging_rate: float
    discharging_rate: float


class NormalizationConfig(BaseModel):
    fit_on: str
    scope: str
    epsilon: float
    interior_only: bool
    interior_trim_frac: float
    apply_on_masked_only: bool
    voltage: Dict[str, Any]
    dvdq: Dict[str, Any]
    d2vdq2: Dict[str, Any]
    dqdv: Dict[str, Any]
    hyst: Dict[str, Any]


class FullConfig(BaseModel):
    GENERAL: General
    HYPER_PARAMETERS: HyperParameters
    PATHS: Paths
    DATASETS: Dict[str, DatasetConfig]
    GP_INTERPOLATION_DATASETS: Dict[str, DatasetConfig] = Field(default_factory=dict)
    NORMALIZATION: NormalizationConfig


class GPExperimentGeneral(BaseModel):
    name: str
    model_type: str
    evaluation_strategy: str
    training: bool = True
    random_seed: int = 42
    holdout_datasets: list[str] = Field(default_factory=list)


class GPAnalysisConfig(BaseModel):
    run_after_training: bool = False
    cycle_bins: int = 10


class GPInterpolationConfig(BaseModel):
    enabled: bool = False
    interpolation_datasets: list[str] = Field(default_factory=list)
    c_rate: Optional[float] = None
    cycle_start: Optional[int] = None
    cycle_end: Optional[int] = None
    use_saved_deployment_models: bool = True
    deployment_outer_test_key: Optional[str] = None
    feature_columns: list[str]
    target_columns: list[str]
    plot_label: Optional[str] = None


class GPRuntimeConfig(BaseModel):
    device: str = "auto"
    use_cuda: bool = True
    dtype: str = "float32"


class GPFeaturesConfig(BaseModel):
    input_columns: list[str]
    target_columns: list[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    group_column: str
    cycle_column: str
    fourier_features: Dict[str, Any] = Field(default_factory=dict)


class GPNormalizationConfig(BaseModel):
    feature_method: str
    feature_column_methods: Dict[str, str] = Field(default_factory=dict)
    target_method: str = "standard"
    target_column_methods: Dict[str, str] = Field(default_factory=dict)


class GPSelectionConfig(BaseModel):
    metric: str = "nll"
    refit_epoch_strategy: str = "median"


class GPModelConfig(BaseModel):
    type: str
    kernel: str
    lengthscale: float
    variance: float
    noise_variance: float
    rq_alpha: float = 0.5
    inducing_points: int
    inducing_method: str
    jitter: float


class GPSchedulerConfig(BaseModel):
    name: str
    mode: str
    factor: float
    patience: int
    threshold: float
    threshold_mode: str
    cooldown: int
    min_lr: float


class GPTrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    min_delta: float
    log_every_epochs: int = 1
    show_progress: bool = False
    scheduler: GPSchedulerConfig


class GPEarlyStoppingConfig(BaseModel):
    enabled: bool = True
    monitor: str = "val_nll"
    patience: int = 8
    min_delta: float = 1e-4
    validation_fraction: float = 0.1


class GPSearchConfig(BaseModel):
    enabled: bool = False
    lengthscale_candidates: list[float] = Field(default_factory=list)
    variance_candidates: list[float] = Field(default_factory=list)
    noise_variance_candidates: list[float] = Field(default_factory=list)
    rq_alpha_candidates: list[float] = Field(default_factory=list)


class GPStudyConfig(BaseModel):
    run_kernel_ablation: bool = False
    kernel_candidates: list[str] = Field(default_factory=list)
    run_feature_normalization_ablation: bool = False
    feature_normalization_candidates: list[str] = Field(default_factory=list)
    run_target_normalization_ablation: bool = False
    target_normalization_candidates: list[str] = Field(default_factory=list)


class GPDatasetConfig(BaseModel):
    id: str
    latent_csv: str
    charging_rate: float
    crate_label: Optional[str] = None


class GPExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    experiment: GPExperimentGeneral
    analysis: GPAnalysisConfig = Field(default_factory=GPAnalysisConfig)
    interpolation: Optional[GPInterpolationConfig] = None
    runtime: GPRuntimeConfig
    features: GPFeaturesConfig
    normalization: GPNormalizationConfig
    selection: GPSelectionConfig
    model: GPModelConfig
    training: GPTrainingConfig
    early_stopping: GPEarlyStoppingConfig = Field(default_factory=GPEarlyStoppingConfig)
    search: GPSearchConfig = Field(default_factory=GPSearchConfig)
    study: GPStudyConfig = Field(default_factory=GPStudyConfig)
    PATHS: Paths
    datasets: Dict[str, GPDatasetConfig]
