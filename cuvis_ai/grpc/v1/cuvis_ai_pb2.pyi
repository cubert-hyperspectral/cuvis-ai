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

class DataConfig(_message.Message):
    __slots__ = ()
    CU3S_FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    ANNOTATION_JSON_PATH_FIELD_NUMBER: _ClassVar[int]
    TRAIN_IDS_FIELD_NUMBER: _ClassVar[int]
    VAL_IDS_FIELD_NUMBER: _ClassVar[int]
    TEST_IDS_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_MODE_FIELD_NUMBER: _ClassVar[int]
    cu3s_file_path: str
    annotation_json_path: str
    train_ids: _containers.RepeatedScalarFieldContainer[int]
    val_ids: _containers.RepeatedScalarFieldContainer[int]
    test_ids: _containers.RepeatedScalarFieldContainer[int]
    batch_size: int
    processing_mode: ProcessingMode
    def __init__(
        self,
        cu3s_file_path: str | None = ...,
        annotation_json_path: str | None = ...,
        train_ids: _Iterable[int] | None = ...,
        val_ids: _Iterable[int] | None = ...,
        test_ids: _Iterable[int] | None = ...,
        batch_size: int | None = ...,
        processing_mode: ProcessingMode | str | None = ...,
    ) -> None: ...

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

class TrainingConfig(_message.Message):
    __slots__ = ()
    CONFIG_JSON_FIELD_NUMBER: _ClassVar[int]
    config_json: bytes
    def __init__(self, config_json: bytes | None = ...) -> None: ...

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

class CreateSessionRequest(_message.Message):
    __slots__ = ()
    PIPELINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    DATA_CONFIG_FIELD_NUMBER: _ClassVar[int]
    pipeline_type: str
    pipeline_config: str
    data_config: DataConfig
    def __init__(
        self,
        pipeline_type: str | None = ...,
        pipeline_config: str | None = ...,
        data_config: DataConfig | _Mapping | None = ...,
    ) -> None: ...

class CreateSessionResponse(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: str | None = ...) -> None: ...

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

class TrainRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINER_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    trainer_type: TrainerType
    config: TrainingConfig
    def __init__(
        self,
        session_id: str | None = ...,
        trainer_type: TrainerType | str | None = ...,
        config: TrainingConfig | _Mapping | None = ...,
    ) -> None: ...

class GetTrainStatusRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: str | None = ...) -> None: ...

class GetTrainStatusResponse(_message.Message):
    __slots__ = ()
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LATEST_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    status: TrainStatus
    latest_progress: TrainResponse
    def __init__(
        self,
        status: TrainStatus | str | None = ...,
        latest_progress: TrainResponse | _Mapping | None = ...,
    ) -> None: ...

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

class ValidateTrainingConfigRequest(_message.Message):
    __slots__ = ()
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    config: TrainingConfig
    def __init__(self, config: TrainingConfig | _Mapping | None = ...) -> None: ...

class ValidateTrainingConfigResponse(_message.Message):
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

class SaveCheckpointRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_PATH_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    checkpoint_path: str
    def __init__(self, session_id: str | None = ..., checkpoint_path: str | None = ...) -> None: ...

class SaveCheckpointResponse(_message.Message):
    __slots__ = ()
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool | None = ..., message: str | None = ...) -> None: ...

class LoadCheckpointRequest(_message.Message):
    __slots__ = ()
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_PATH_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    checkpoint_path: str
    def __init__(self, session_id: str | None = ..., checkpoint_path: str | None = ...) -> None: ...

class LoadCheckpointResponse(_message.Message):
    __slots__ = ()
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool | None = ..., message: str | None = ...) -> None: ...
