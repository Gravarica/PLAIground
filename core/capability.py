from abc import ABC, abstractmethod
from typing import Dict, Any

class HandlerError(Exception):
    pass


class CapabilityHandler(ABC):

    @abstractmethod
    def execute(self, connector: Any, model_id: str, config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

class LLMHandler(CapabilityHandler):

    def execute(self, connector: Any, model_id: str, config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:

        prompt_template = config.get("parameters", {}).get("prompt")
        prompt = prompt_template.format(**input_data) if prompt_template else input_data.get("prompt", str(input_data))

        # Backward compatibility
        if hasattr(connector, 'generate'):
            result = connector.generate(model_id, prompt, **config)
            return {"output": result}

        return connector.execute({"prompt": prompt, **config})

class ObjectDetectionHandler(CapabilityHandler):

    def execute(self, connector: Any, model_id: str, config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:

        # This is specificity to certain connector implementations
        if hasattr(connector, '_current_model'):
            connector._current_model = model_id

        # Backward compatibility
        if hasattr(connector, 'detect'):
            image = input_data.get("image", "")
            detections = connector.detect(model_id, image, **config)
        else:
            result = connector.execute(input_data)
            detections = result.get("detections", [])

        has_detection = len(detections) > 0
        return {
            "detections": detections,
            "has_detection": has_detection,
            "output": "detected" if has_detection else "none"
        }

class CapabilityRegistry:

    _handlers: Dict[str, CapabilityHandler] = {
        "llm": LLMHandler(),
        "object_detection": ObjectDetectionHandler(),
    }

    @classmethod
    def register(cls, capability:str, handler:CapabilityHandler) -> None:
        cls._handlers[capability] = handler

    @classmethod
    def get_handler(cls, capability:str) -> CapabilityHandler:
        handler = cls._handlers.get(capability)
        if not handler:
            raise HandlerError("Unknown capability {}".format(capability))

        return handler