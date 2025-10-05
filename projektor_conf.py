import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
from dataclasses import dataclass

@dataclass(kw_only=True)
class Configuration:
	"""The configurable fields for the chatbot."""
	user_id: str = "default-user" 
	tasks_category: str = "software_project" 
	projektor_role: str = (
		"Eres Projektor, un asistente de ingeniería de software. Te encargas de llevar el registro "
		"de la información sobre el proyecto, y también de crear, organizar y gestionar la lista de"
		" tareas por completar del proyecto, así como también de los metadatos del proyecto y el "
		"usuario. "
	)

	@classmethod
	def from_runnable_config(
		cls, config: Optional[RunnableConfig] = None
	) -> "Configuration":
		"""Create a Configuration instance from a RunnableConfig."""
		configurable = (
			config["configurable"] if config and "configurable" in config else {}
		)
		values: dict[str, Any] = {
			f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
			for f in fields(cls)
			if f.init
		}
		return cls(**{k: v for k, v in values.items() if v})