from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from .capability import CapabilityRegistry

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

        if not self.is_bound():
            raise RuntimeError(f"Model '{self.name}' not bound. Call registry.bind() first.")


        if self.input_contract and not self.input_contract.validate(input_data):
            raise ValueError(f"Input validation failed for '{self.name}'")

        capability = self._get_capability()
        handler = CapabilityRegistry.get_handler(capability)

        task_config = self.task_contract.config.parameters if self.task_contract else {}
        effective = {**task_config, **self._config}

        output = handler.execute(self._connector, self._model_id, effective, input_data)

        if self.output_contract and not self.output_contract.validate(output):
            raise ValueError(f"Output validation failed for '{self.name}'")

        return output
