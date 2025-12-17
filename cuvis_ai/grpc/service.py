"""gRPC service implementation delegating to modular components."""

from __future__ import annotations

from collections.abc import Iterable

from .config_service import ConfigService
from .discovery_service import DiscoveryService
from .inference_service import InferenceService
from .introspection_service import IntrospectionService
from .pipeline_service import PipelineService
from .session_manager import SessionManager
from .session_service import SessionService
from .training_service import TrainingService
from .trainrun_service import TrainRunService
from .v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc


class CuvisAIService(cuvis_ai_pb2_grpc.CuvisAIServiceServicer):
    """Main gRPC service class delegating to specialized service components."""

    def __init__(self, session_manager: SessionManager | None = None) -> None:
        self.session_manager = session_manager or SessionManager()

        # Initialize service components
        self.session_service = SessionService(self.session_manager)
        self.config_service = ConfigService(self.session_manager)
        self.pipeline_service = PipelineService(self.session_manager)
        self.inference_service = InferenceService(self.session_manager)
        self.training_service = TrainingService(self.session_manager)
        self.trainrun_service = TrainRunService(self.session_manager)
        self.discovery_service = DiscoveryService()
        self.introspection_service = IntrospectionService(self.session_manager)

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------
    def CreateSession(self, request, context) -> cuvis_ai_pb2.CreateSessionResponse:
        return self.session_service.create_session(request, context)

    def CloseSession(self, request, context) -> cuvis_ai_pb2.CloseSessionResponse:
        return self.session_service.close_session(request, context)

    def SetSessionSearchPaths(self, request, context) -> cuvis_ai_pb2.SetSessionSearchPathsResponse:
        return self.session_service.set_session_search_paths(request, context)

    # ------------------------------------------------------------------
    # Config resolution and validation
    # ------------------------------------------------------------------
    def ResolveConfig(self, request, context) -> cuvis_ai_pb2.ResolveConfigResponse:
        return self.config_service.resolve_config(request, context)

    def GetParameterSchema(self, request, context) -> cuvis_ai_pb2.GetParameterSchemaResponse:
        return self.config_service.get_parameter_schema(request, context)

    def ValidateConfig(self, request, context) -> cuvis_ai_pb2.ValidateConfigResponse:
        return self.config_service.validate_config(request, context)

    # ------------------------------------------------------------------
    # Pipeline construction and training configuration
    # ------------------------------------------------------------------

    def LoadPipelineWeights(self, request, context) -> cuvis_ai_pb2.LoadPipelineWeightsResponse:
        return self.pipeline_service.load_pipeline_weights(request, context)

    def SetTrainRunConfig(self, request, context) -> cuvis_ai_pb2.SetTrainRunConfigResponse:
        return self.pipeline_service.set_train_run_config(request, context)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def Inference(self, request, context) -> cuvis_ai_pb2.InferenceResponse:
        return self.inference_service.inference(request, context)

    # ------------------------------------------------------------------
    # Pipeline Introspection
    # ------------------------------------------------------------------
    def GetPipelineInputs(self, request, context) -> cuvis_ai_pb2.GetPipelineInputsResponse:
        return self.introspection_service.get_pipeline_inputs(request, context)

    def GetPipelineOutputs(self, request, context) -> cuvis_ai_pb2.GetPipelineOutputsResponse:
        return self.introspection_service.get_pipeline_outputs(request, context)

    def GetPipelineVisualization(
        self, request, context
    ) -> cuvis_ai_pb2.GetPipelineVisualizationResponse:
        return self.introspection_service.get_pipeline_visualization(request, context)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def Train(self, request, context) -> Iterable[cuvis_ai_pb2.TrainResponse]:
        return self.training_service.train(request, context)

    def GetTrainStatus(self, request, context) -> cuvis_ai_pb2.GetTrainStatusResponse:
        return self.training_service.get_train_status(request, context)

    # ------------------------------------------------------------------
    # Pipeline Management (Model Deployment)
    # ------------------------------------------------------------------
    def SavePipeline(self, request, context) -> cuvis_ai_pb2.SavePipelineResponse:
        return self.pipeline_service.save_pipeline(request, context)

    def LoadPipeline(self, request, context) -> cuvis_ai_pb2.LoadPipelineResponse:
        return self.pipeline_service.load_pipeline(request, context)

    # ------------------------------------------------------------------
    # Experiment Management (Reproducibility)
    # ------------------------------------------------------------------
    def SaveTrainRun(self, request, context) -> cuvis_ai_pb2.SaveTrainRunResponse:
        return self.trainrun_service.save_train_run(request, context)

    def RestoreTrainRun(self, request, context) -> cuvis_ai_pb2.RestoreTrainRunResponse:
        return self.trainrun_service.restore_train_run(request, context)

    # ------------------------------------------------------------------
    # Pipeline Discovery
    # ------------------------------------------------------------------
    def ListAvailablePipelinees(
        self, request, context
    ) -> cuvis_ai_pb2.ListAvailablePipelineesResponse:
        return self.discovery_service.list_available_pipelinees(request, context)

    def GetPipelineInfo(self, request, context) -> cuvis_ai_pb2.GetPipelineInfoResponse:
        return self.discovery_service.get_pipeline_info(request, context)

    # ------------------------------------------------------------------
    # Training Capabilities
    # ------------------------------------------------------------------
    def GetTrainingCapabilities(
        self, request, context
    ) -> cuvis_ai_pb2.GetTrainingCapabilitiesResponse:
        return self.training_service.get_training_capabilities(request, context)


__all__ = ["CuvisAIService"]
