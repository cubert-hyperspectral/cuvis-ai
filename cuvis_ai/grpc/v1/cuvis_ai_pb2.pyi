from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class ProcessingMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROCESSING_MODE_UNSPECIFIED: _ClassVar[ProcessingMode]
    PROCESSING_MODE_RAW: _ClassVar[ProcessingMode]
    PROCESSING_MODE_REFLECTANCE: _ClassVar[ProcessingMode]
    PROCESSING_MODE_DARKSUBTRACT: _ClassVar[ProcessingMode]
    PROCESSING_MODE_SPECTRAL_RADIANCE: _ClassVar[ProcessingMode]

class ExecutionStage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXECUTION_STAGE_UNSPECIFIED: _ClassVar[ExecutionStage]
    EXECUTION_STAGE_TRAIN: _ClassVar[ExecutionStage]
    EXECUTION_STAGE_VAL: _ClassVar[ExecutionStage]
    EXECUTION_STAGE_TEST: _ClassVar[ExecutionStage]
    EXECUTION_STAGE_INFERENCE: _ClassVar[ExecutionStage]

class DType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    D_TYPE_UNSPECIFIED: _ClassVar[DType]
    D_TYPE_FLOAT32: _ClassVar[DType]
    D_TYPE_FLOAT64: _ClassVar[DType]
    D_TYPE_INT32: _ClassVar[DType]
    D_TYPE_INT64: _ClassVar[DType]
    D_TYPE_UINT8: _ClassVar[DType]
    D_TYPE_BOOL: _ClassVar[DType]
    D_TYPE_FLOAT16: _ClassVar[DType]
    D_TYPE_UINT16: _ClassVar[DType]

class TrainerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRAINER_TYPE_UNSPECIFIED: _ClassVar[TrainerType]
    TRAINER_TYPE_STATISTICAL: _ClassVar[TrainerType]
    TRAINER_TYPE_GRADIENT: _ClassVar[TrainerType]

class TrainStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRAIN_STATUS_UNSPECIFIED: _ClassVar[TrainStatus]
    TRAIN_STATUS_RUNNING: _ClassVar[TrainStatus]
    TRAIN_STATUS_COMPLETE: _ClassVar[TrainStatus]
    TRAIN_STATUS_ERROR: _ClassVar[TrainStatus]

class PointType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    POINT_TYPE_UNSPECIFIED: _ClassVar[PointType]
    POINT_TYPE_POSITIVE: _ClassVar[PointType]
    POINT_TYPE_NEGATIVE: _ClassVar[PointType]
    POINT_TYPE_NEUTRAL: _ClassVar[PointType]

PROCESSING_MODE_UNSPECIFIED: ProcessingMode
PROCESSING_MODE_RAW: ProcessingMode
PROCESSING_MODE_REFLECTANCE: ProcessingMode
PROCESSING_MODE_DARKSUBTRACT: ProcessingMode
PROCESSING_MODE_SPECTRAL_RADIANCE: ProcessingMode
EXECUTION_STAGE_UNSPECIFIED: ExecutionStage
EXECUTION_STAGE_TRAIN: ExecutionStage
EXECUTION_STAGE_VAL: ExecutionStage
EXECUTION_STAGE_TEST: ExecutionStage
EXECUTION_STAGE_INFERENCE: ExecutionStage
D_TYPE_UNSPECIFIED: DType
D_TYPE_FLOAT32: DType
D_TYPE_FLOAT64: DType
D_TYPE_INT32: DType
D_TYPE_INT64: DType
D_TYPE_UINT8: DType
D_TYPE_BOOL: DType
D_TYPE_FLOAT16: DType
D_TYPE_UINT16: DType
TRAINER_TYPE_UNSPECIFIED: TrainerType
TRAINER_TYPE_STATISTICAL: TrainerType
TRAINER_TYPE_GRADIENT: TrainerType
TRAIN_STATUS_UNSPECIFIED: TrainStatus
TRAIN_STATUS_RUNNING: TrainStatus
TRAIN_STATUS_COMPLETE: TrainStatus
TRAIN_STATUS_ERROR: TrainStatus
POINT_TYPE_UNSPECIFIED: PointType
POINT_TYPE_POSITIVE: PointType
POINT_TYPE_NEGATIVE: PointType
POINT_TYPE_NEUTRAL: PointType

class Tensor(_message.Message):
    __slots__ = ()
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: DType
    raw_data: bytes
    def __init__(
        self,
        shape: _Iterable[int] | None = ...,
        dtype: DType | str | None = ...,
        raw_data: bytes | None = ...,
    ) -> None: ...

class Context(_message.Message):
    __slots__ = ()
    STAGE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    BATCH_IDX_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_STEP_FIELD_NUMBER: _ClassVar[int]
    stage: ExecutionStage
    epoch: int
    batch_idx: int
    global_step: int
    def __init__(
        self,
        stage: ExecutionStage | str | None = ...,
        epoch: int | None = ...,
        batch_idx: int | None = ...,
        global_step: int | None = ...,
    ) -> None: ...

class PipelineConfig(_message.Message):
    __slots__ = ()
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: bytes | None = ...) -> None: ...

class DataConfig(_message.Message):
    __slots__ = ()
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: bytes | None = ...) -> None: ...

class OptimizerConfig(_message.Message):
    __slots__ = ()
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: bytes | None = ...) -> None: ...

class SchedulerConfig(_message.Message):
    __slots__ = ()
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: bytes | None = ...) -> None: ...

class CallbacksConfig(_message.Message):
    __slots__ = ()
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: bytes | None = ...) -> None: ...

class PipelineMetadata(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    CREATED_FIELD_NUMBER: _ClassVar[int]
    CUVIS_AI_VERSION_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    created: str
    cuvis_ai_version: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    author: str
    def __init__(
        self,
        name: str | None = ...,
        description: str | None = ...,
        created: str | None = ...,
        cuvis_ai_version: str | None = ...,
        tags: _Iterable[str] | None = ...,
        author: str | None = ...,
    ) -> None: ...

class PipelineInfo(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    HAS_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_PATH_FIELD_NUMBER: _ClassVar[int]
    YAML_CONTENT_FIELD_NUMBER: _ClassVar[int]
    name: str
    path: str
    metadata: PipelineMetadata
    tags: _containers.RepeatedScalarFieldContainer[str]
    has_weights: bool
    weights_path: str
    yaml_content: str
    def __init__(
        self,
        name: str | None = ...,
        path: str | None = ...,
        metadata: PipelineMetadata | _Mapping | None = ...,
        tags: _Iterable[str] | None = ...,
        has_weights: bool | None = ...,
        weights_path: str | None = ...,
        yaml_content: str | None = ...,
    ) -> None: ...

class TrainingConfig(_message.Message):
    __slots__ = ()
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: bytes | None = ...) -> None: ...

class TrainRunConfig(_message.Message):
    __slots__ = ()
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: bytes | None = ...) -> None: ...

class BoundingBox(_message.Message):
    __slots__ = ()
    ELEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    X_MIN_FIELD_NUMBER: _ClassVar[int]
    Y_MIN_FIELD_NUMBER: _ClassVar[int]
    X_MAX_FIELD_NUMBER: _ClassVar[int]
    Y_MAX_FIELD_NUMBER: _ClassVar[int]
    element_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    def __init__(
        self,
        element_id: int | None = ...,
        x_min: float | None = ...,
        y_min: float | None = ...,
        x_max: float | None = ...,
        y_max: float | None = ...,
    ) -> None: ...

class BoundingBoxes(_message.Message):
    __slots__ = ()
    BOXES_FIELD_NUMBER: _ClassVar[int]
    boxes: _containers.RepeatedCompositeFieldContainer[BoundingBox]
    def __init__(self, boxes: _Iterable[BoundingBox | _Mapping] | None = ...) -> None: ...

class Point(_message.Message):
    __slots__ = ()
    ELEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    element_id: int
    x: float
    y: float
    type: PointType
    def __init__(
        self,
        element_id: int | None = ...,
        x: float | None = ...,
        y: float | None = ...,
        type: PointType | str | None = ...,
    ) -> None: ...

class Points(_message.Message):
    __slots__ = ()
    POINTS_FIELD_NUMBER: _ClassVar[int]
    points: _containers.RepeatedCompositeFieldContainer[Point]
    def __init__(self, points: _Iterable[Point | _Mapping] | None = ...) -> None: ...

class InputBatch(_message.Message):
    __slots__ = ()
    class ExtraInputsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tensor
        def __init__(
            self, key: str | None = ..., value: Tensor | _Mapping | None = ...
        ) -> None: ...

    WAVELENGTHS_FIELD_NUMBER: _ClassVar[int]
    CUBE_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    BBOXES_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    TEXT_PROMPT_FIELD_NUMBER: _ClassVar[int]
    EXTRA_INPUTS_FIELD_NUMBER: _ClassVar[int]
    wavelengths: Tensor
    cube: Tensor
    mask: Tensor
    bboxes: BoundingBoxes
    points: Points
    text_prompt: str
    extra_inputs: _containers.MessageMap[str, Tensor]
    def __init__(
        self,
        wavelengths: Tensor | _Mapping | None = ...,
        cube: Tensor | _Mapping | None = ...,
        mask: Tensor | _Mapping | None = ...,
        bboxes: BoundingBoxes | _Mapping | None = ...,
        points: Points | _Mapping | None = ...,
        text_prompt: str | None = ...,
        extra_inputs: _Mapping[str, Tensor] | None = ...,
    ) -> None: ...

class TensorSpec(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: DType
    required: bool
    def __init__(
        self,
        name: str | None = ...,
        shape: _Iterable[int] | None = ...,
        dtype: DType | str | None = ...,
        required: bool | None = ...,
    ) -> None: ...

class TrainResponse(_message.Message):
    __slots__ = ()
    class LossesEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: str | None = ..., value: float | None = ...) -> None: ...

    class MetricsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: str | None = ..., value: float | None = ...) -> None: ...

    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    LOSSES_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    context: Context
    losses: _containers.ScalarMap[str, float]
    metrics: _containers.ScalarMap[str, float]
    status: TrainStatus
    message: str
    def __init__(
        self,
        context: Context | _Mapping | None = ...,
        losses: _Mapping[str, float] | None = ...,
        metrics: _Mapping[str, float] | None = ...,
        status: TrainStatus | str | None = ...,
        message: str | None = ...,
    ) -> None: ...

class ParamSpec(_message.Message):
    __slots__ = ()
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    required: bool
    default_value: str
    description: str
    validation: str
    def __init__(
        self,
        name: str | None = ...,
        type: str | None = ...,
        required: bool | None = ...,
        default_value: str | None = ...,
        description: str | None = ...,
        validation: str | None = ...,
    ) -> None: ...

class CallbackTypeInfo(_message.Message):
    __slots__ = ()
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    type: str
    description: str
    parameters: _containers.RepeatedCompositeFieldContainer[ParamSpec]
    def __init__(
        self,
        type: str | None = ...,
        description: str | None = ...,
        parameters: _Iterable[ParamSpec | _Mapping] | None = ...,
    ) -> None: ...

class OptimizerParamsSchema(_message.Message):
    __slots__ = ()
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    parameters: _containers.RepeatedCompositeFieldContainer[ParamSpec]
    def __init__(self, parameters: _Iterable[ParamSpec | _Mapping] | None = ...) -> None: ...

class SchedulerParamsSchema(_message.Message):
    __slots__ = ()
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    parameters: _containers.RepeatedCompositeFieldContainer[ParamSpec]
    def __init__(self, parameters: _Iterable[ParamSpec | _Mapping] | None = ...) -> None: ...

class ListAvailablePipelineesRequest(_message.Message):
    __slots__ = ()
    FILTER_TAG_FIELD_NUMBER: _ClassVar[int]
    filter_tag: str
    def __init__(self, filter_tag: str | None = ...) -> None: ...

class ListAvailablePipelineesResponse(_message.Message):
    __slots__ = ()
    PIPELINEES_FIELD_NUMBER: _ClassVar[int]
    pipelinees: _containers.RepeatedCompositeFieldContainer[PipelineInfo]
    def __init__(self, pipelinees: _Iterable[PipelineInfo | _Mapping] | None = ...) -> None: ...

class GetPipelineInfoRequest(_message.Message):
    __slots__ = ()
    PIPELINE_NAME_FIELD_NUMBER: _ClassVar[int]
    pipeline_name: str
    def __init__(self, pipeline_name: str | None = ...) -> None: ...

class GetPipelineInfoResponse(_message.Message):
    __slots__ = ()
    PIPELINE_INFO_FIELD_NUMBER: _ClassVar[int]
    pipeline_info: PipelineInfo
    def __init__(self, pipeline_info: PipelineInfo | _Mapping | None = ...) -> None: ...

class CreateSessionRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CreateSessionResponse(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: str | None = ...) -> None: ...

class SetSessionSearchPathsRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SEARCH_PATHS_FIELD_NUMBER: _ClassVar[int]
    APPEND_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    search_paths: _containers.RepeatedScalarFieldContainer[str]
    append: bool
    def __init__(
        self,
        session_id: str | None = ...,
        search_paths: _Iterable[str] | None = ...,
        append: bool | None = ...,
    ) -> None: ...

class SetSessionSearchPathsResponse(_message.Message):
    __slots__ = ()
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_PATHS_FIELD_NUMBER: _ClassVar[int]
    REJECTED_PATHS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    current_paths: _containers.RepeatedScalarFieldContainer[str]
    rejected_paths: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        success: bool | None = ...,
        current_paths: _Iterable[str] | None = ...,
        rejected_paths: _Iterable[str] | None = ...,
    ) -> None: ...

class CloseSessionRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: str | None = ...) -> None: ...

class CloseSessionResponse(_message.Message):
    __slots__ = ()
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool | None = ...) -> None: ...

class ResolveConfigRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_TYPE_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    OVERRIDES_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    config_type: str
    path: str
    overrides: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        session_id: str | None = ...,
        config_type: str | None = ...,
        path: str | None = ...,
        overrides: _Iterable[str] | None = ...,
    ) -> None: ...

class ResolveConfigResponse(_message.Message):
    __slots__ = ()
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: bytes | None = ...) -> None: ...

class GetParameterSchemaRequest(_message.Message):
    __slots__ = ()
    CONFIG_TYPE_FIELD_NUMBER: _ClassVar[int]
    config_type: str
    def __init__(self, config_type: str | None = ...) -> None: ...

class GetParameterSchemaResponse(_message.Message):
    __slots__ = ()
    JSON_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    json_schema: str
    def __init__(self, json_schema: str | None = ...) -> None: ...

class ValidateConfigRequest(_message.Message):
    __slots__ = ()
    CONFIG_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_type: str
    config_bytes: bytes
    def __init__(self, config_type: str | None = ..., config_bytes: bytes | None = ...) -> None: ...

class ValidateConfigResponse(_message.Message):
    __slots__ = ()
    VALID_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    valid: bool
    errors: _containers.RepeatedScalarFieldContainer[str]
    warnings: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        valid: bool | None = ...,
        errors: _Iterable[str] | None = ...,
        warnings: _Iterable[str] | None = ...,
    ) -> None: ...

class LoadPipelineWeightsRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_PATH_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_BYTES_FIELD_NUMBER: _ClassVar[int]
    STRICT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    weights_path: str
    weights_bytes: bytes
    strict: bool
    def __init__(
        self,
        session_id: str | None = ...,
        weights_path: str | None = ...,
        weights_bytes: bytes | None = ...,
        strict: bool | None = ...,
    ) -> None: ...

class LoadPipelineWeightsResponse(_message.Message):
    __slots__ = ()
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_PATH_FIELD_NUMBER: _ClassVar[int]
    success: bool
    resolved_path: str
    def __init__(self, success: bool | None = ..., resolved_path: str | None = ...) -> None: ...

class SetTrainRunConfigRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    config: TrainRunConfig
    def __init__(
        self,
        session_id: str | None = ...,
        config: TrainRunConfig | _Mapping | None = ...,
    ) -> None: ...

class SetTrainRunConfigResponse(_message.Message):
    __slots__ = ()
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_FROM_CONFIG_FIELD_NUMBER: _ClassVar[int]
    success: bool
    pipeline_from_config: bool
    def __init__(
        self, success: bool | None = ..., pipeline_from_config: bool | None = ...
    ) -> None: ...

class TrainRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINER_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    TRAINING_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    trainer_type: TrainerType
    data: DataConfig
    training: TrainingConfig
    def __init__(
        self,
        session_id: str | None = ...,
        trainer_type: TrainerType | str | None = ...,
        data: DataConfig | _Mapping | None = ...,
        training: TrainingConfig | _Mapping | None = ...,
    ) -> None: ...

class GetTrainStatusRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: str | None = ...) -> None: ...

class GetTrainStatusResponse(_message.Message):
    __slots__ = ()
    LATEST_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    latest_progress: TrainResponse
    def __init__(self, latest_progress: TrainResponse | _Mapping | None = ...) -> None: ...

class GetTrainingCapabilitiesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetTrainingCapabilitiesResponse(_message.Message):
    __slots__ = ()
    SUPPORTED_OPTIMIZERS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_SCHEDULERS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_CALLBACKS_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_PARAMS_FIELD_NUMBER: _ClassVar[int]
    SCHEDULER_PARAMS_FIELD_NUMBER: _ClassVar[int]
    supported_optimizers: _containers.RepeatedScalarFieldContainer[str]
    supported_schedulers: _containers.RepeatedScalarFieldContainer[str]
    supported_callbacks: _containers.RepeatedCompositeFieldContainer[CallbackTypeInfo]
    optimizer_params: OptimizerParamsSchema
    scheduler_params: SchedulerParamsSchema
    def __init__(
        self,
        supported_optimizers: _Iterable[str] | None = ...,
        supported_schedulers: _Iterable[str] | None = ...,
        supported_callbacks: _Iterable[CallbackTypeInfo | _Mapping] | None = ...,
        optimizer_params: OptimizerParamsSchema | _Mapping | None = ...,
        scheduler_params: SchedulerParamsSchema | _Mapping | None = ...,
    ) -> None: ...

class SavePipelineRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_PATH_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    pipeline_path: str
    metadata: PipelineMetadata
    def __init__(
        self,
        session_id: str | None = ...,
        pipeline_path: str | None = ...,
        metadata: PipelineMetadata | _Mapping | None = ...,
    ) -> None: ...

class SavePipelineResponse(_message.Message):
    __slots__ = ()
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_PATH_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_PATH_FIELD_NUMBER: _ClassVar[int]
    success: bool
    pipeline_path: str
    weights_path: str
    def __init__(
        self,
        success: bool | None = ...,
        pipeline_path: str | None = ...,
        weights_path: str | None = ...,
    ) -> None: ...

class LoadPipelineRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    pipeline: PipelineConfig
    def __init__(
        self,
        session_id: str | None = ...,
        pipeline: PipelineConfig | _Mapping | None = ...,
    ) -> None: ...

class LoadPipelineResponse(_message.Message):
    __slots__ = ()
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    success: bool
    metadata: PipelineMetadata
    def __init__(
        self,
        success: bool | None = ...,
        metadata: PipelineMetadata | _Mapping | None = ...,
    ) -> None: ...

class SaveTrainRunRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINRUN_PATH_FIELD_NUMBER: _ClassVar[int]
    SAVE_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    trainrun_path: str
    save_weights: bool
    def __init__(
        self,
        session_id: str | None = ...,
        trainrun_path: str | None = ...,
        save_weights: bool | None = ...,
    ) -> None: ...

class SaveTrainRunResponse(_message.Message):
    __slots__ = ()
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    TRAINRUN_PATH_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_PATH_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_PATH_FIELD_NUMBER: _ClassVar[int]
    success: bool
    trainrun_path: str
    pipeline_path: str
    weights_path: str
    def __init__(
        self,
        success: bool | None = ...,
        trainrun_path: str | None = ...,
        pipeline_path: str | None = ...,
        weights_path: str | None = ...,
    ) -> None: ...

class RestoreTrainRunRequest(_message.Message):
    __slots__ = ()
    TRAINRUN_PATH_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_PATH_FIELD_NUMBER: _ClassVar[int]
    STRICT_FIELD_NUMBER: _ClassVar[int]
    trainrun_path: str
    weights_path: str
    strict: bool
    def __init__(
        self,
        trainrun_path: str | None = ...,
        weights_path: str | None = ...,
        strict: bool | None = ...,
    ) -> None: ...

class RestoreTrainRunResponse(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINRUN_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    trainrun: TrainRunConfig
    def __init__(
        self,
        session_id: str | None = ...,
        trainrun: TrainRunConfig | _Mapping | None = ...,
    ) -> None: ...

class GetPipelineInputsRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: str | None = ...) -> None: ...

class GetPipelineInputsResponse(_message.Message):
    __slots__ = ()
    class InputSpecsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: TensorSpec
        def __init__(
            self, key: str | None = ..., value: TensorSpec | _Mapping | None = ...
        ) -> None: ...

    INPUT_NAMES_FIELD_NUMBER: _ClassVar[int]
    INPUT_SPECS_FIELD_NUMBER: _ClassVar[int]
    input_names: _containers.RepeatedScalarFieldContainer[str]
    input_specs: _containers.MessageMap[str, TensorSpec]
    def __init__(
        self,
        input_names: _Iterable[str] | None = ...,
        input_specs: _Mapping[str, TensorSpec] | None = ...,
    ) -> None: ...

class GetPipelineOutputsRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: str | None = ...) -> None: ...

class GetPipelineOutputsResponse(_message.Message):
    __slots__ = ()
    class OutputSpecsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: TensorSpec
        def __init__(
            self, key: str | None = ..., value: TensorSpec | _Mapping | None = ...
        ) -> None: ...

    OUTPUT_NAMES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SPECS_FIELD_NUMBER: _ClassVar[int]
    output_names: _containers.RepeatedScalarFieldContainer[str]
    output_specs: _containers.MessageMap[str, TensorSpec]
    def __init__(
        self,
        output_names: _Iterable[str] | None = ...,
        output_specs: _Mapping[str, TensorSpec] | None = ...,
    ) -> None: ...

class GetPipelineVisualizationRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    format: str
    def __init__(self, session_id: str | None = ..., format: str | None = ...) -> None: ...

class GetPipelineVisualizationResponse(_message.Message):
    __slots__ = ()
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    image_data: bytes
    format: str
    def __init__(self, image_data: bytes | None = ..., format: str | None = ...) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SPECS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    inputs: InputBatch
    output_specs: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        session_id: str | None = ...,
        inputs: InputBatch | _Mapping | None = ...,
        output_specs: _Iterable[str] | None = ...,
    ) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ()
    class OutputsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tensor
        def __init__(
            self, key: str | None = ..., value: Tensor | _Mapping | None = ...
        ) -> None: ...

    class MetricsEntry(_message.Message):
        __slots__ = ()
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: str | None = ..., value: float | None = ...) -> None: ...

    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    outputs: _containers.MessageMap[str, Tensor]
    metrics: _containers.ScalarMap[str, float]
    def __init__(
        self,
        outputs: _Mapping[str, Tensor] | None = ...,
        metrics: _Mapping[str, float] | None = ...,
    ) -> None: ...
