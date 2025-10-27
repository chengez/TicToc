from abc import ABC, abstractmethod

class Base_Handler(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def format_input(self, history, *args, **kwargs):
        """Format the input history for the model."""
        pass

    @abstractmethod
    def run_inference(self, formatted_inputs):
        """Run inference on the formatted inputs (batch or single)."""
        pass
