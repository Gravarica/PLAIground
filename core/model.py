from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..contracts.data_contract import DataContract
    from ..contracts.task_contract import TaskContract
    from ..connectors.base import LLMConnector


@dataclass
class CompoundableModel:
    """Compoundable Model encapsulating the Three Contracts."""

    name: str
    input_contract: Optional['DataContract'] = None
    output_contract: Optional['DataContract'] = None
    task_contract: Optional['TaskContract'] = None

    _connector: Optional['LLMConnector'] = field(default=None, repr=False)
    _model_id: Optional[str] = field(default=None, repr=False)
    _config: Dict[str, Any] = field(default_factory=dict, repr=False)

    def bind(self, connector: 'LLMConnector', model_id: str, **config):
        """Bind this model to a specific implementation."""
        self._connector = connector
        self._model_id = model_id
        self._config = config

    def is_bound(self) -> bool:
        return self._connector is not None and self._model_id is not None

    def _get_prompt_template(self) -> Optional[str]:
        """Get prompt template from Task Configuration."""
        if self.task_contract and self.task_contract.config:
            return self.task_contract.config.parameters.get("prompt")
        return None

    def _format_prompt(self, input_data: Dict[str, Any]) -> str:
        """Format prompt using template from Task Configuration."""
        prompt_template = self._get_prompt_template()
        if prompt_template:
            return prompt_template.format(**input_data)
        return input_data.get("prompt", str(input_data))

    def _get_capability(self) -> str:
        """Get capability from Task Contract."""
        if self.task_contract:
            return self.task_contract.capability
        return "llm"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the compoundable model."""

        if not self.is_bound():
            raise RuntimeError(f"Model '{self.name}' not bound. Call registry.bind() first.")


        if self.input_contract and not self.input_contract.validate(input_data):
            raise ValueError(f"Input validation failed for '{self.name}'")

        capability = self._get_capability()

        if capability == "object_detection":
            output = self._execute_object_detection(input_data)
        else:
            output = self._execute_llm(input_data)

        if self.output_contract and not self.output_contract.validate(output):
            raise ValueError(f"Output validation failed for '{self.name}'")

        return output

    def _execute_llm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM/classification models."""
        prompt = self._format_prompt(input_data)
        result = self._connector.generate(self._model_id, prompt, **self._config)
        return {"output": result}

    def _execute_object_detection(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute object detection models via generic connector.

        The connector is infrastructure-agnostic - it just knows how to
        talk to Triton/TorchServe/etc. Uses the translation pipeline.
        """
        # Set current model on connector (for translation pipeline)
        self._connector._current_model = self._model_id

        # Use detect() for backward compatibility (calls execute internally)
        if hasattr(self._connector, 'detect'):
            image = input_data.get("image", "")
            detections = self._connector.detect(self._model_id, image, **self._config)
            has_detection = len(detections) > 0
            return {
                "detections": detections,
                "has_detection": has_detection,
                "output": "detected" if has_detection else "none"
            }
        # Direct execute() call with new pipeline
        elif hasattr(self._connector, 'execute'):
            result = self._connector.execute(input_data)
            detections = result.get("detections", [])
            has_detection = result.get("has_detection", len(detections) > 0)
            return {
                "detections": detections,
                "has_detection": has_detection,
                "output": "detected" if has_detection else "none"
            }
        else:
            raise RuntimeError(f"Connector does not support object detection")
