import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from trustcall import create_extractor

from typing import Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

import uuid
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

import projektor_conf

## Utilities 

# Inspect the tool calls for Trustcall
class Spy:
	def __init__(self):
		self.called_tools = []

	def __call__(self, run):
		q = [run]
		while q:
			r = q.pop()
			if r.child_runs:
				q.extend(r.child_runs)
			if r.run_type == "chat_model":
				self.called_tools.append(
					r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
				)

# Extract information from tool calls for both patches and new memories in Trustcall
def extract_tool_info(tool_calls, schema_name="Memory"):
	"""Extract information from tool calls for both patches and new memories.
	
	Args:
		tool_calls: List of tool calls from the model
		schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Project")
	"""
	# Initialize list of changes
	changes = []
	
	for call_group in tool_calls:
		for call in call_group:
			if call['name'] == 'PatchDoc':
				# Check if there are any patches
				if call['args']['patches']:
					changes.append({
						'type': 'update',
						'doc_id': call['args']['json_doc_id'],
						'planned_edits': call['args']['planned_edits'],
						'value': call['args']['patches'][0]['value']
					})
				else:
					# Handle case where no changes were needed
					changes.append({
						'type': 'no_update',
						'doc_id': call['args']['json_doc_id'],
						'planned_edits': call['args']['planned_edits']
					})
			elif call['name'] == schema_name:
				changes.append({
					'type': 'new',
					'value': call['args']
				})

	# Format results as a single string
	result_parts = []
	for change in changes:
		if change['type'] == 'update':
			result_parts.append(
				f"Document {change['doc_id']} updated:\n"
				f"Plan: {change['planned_edits']}\n"
				f"Added content: {change['value']}"
			)
		elif change['type'] == 'no_update':
			result_parts.append(
				f"Document {change['doc_id']} unchanged:\n"
				f"{change['planned_edits']}"
			)
		else:
			result_parts.append(
				f"New {schema_name} created:\n"
				f"Content: {change['value']}"
			)
	
	return "\n\n".join(result_parts)

## Schema definitions
class PerfilDesarrollador(BaseModel):
	"""Información básica sobre el perfil del desarrollador"""

	nombre: str = Field(
		description="Nombre completo del desarrollador"
	)
	nivel_experiencia: Literal["principiante", "junior", "intermedio"] = Field(
		description="Nivel actual de experiencia del desarrollador",
		default="principiante"
	)
	tecnologias_disponibles: list[str] = Field(
		description="Lista de tecnologías o herramientas prominentes que el desarrollador sabe usar",
		default_factory=list
	)
	lenguaje_preferido: Optional[str] = Field(
		description="Lenguaje de programación preferido (ej: Java, Python, TypeScript)",
		default=None
	)


class MetadatosProyecto(BaseModel):
	"""Metadatos generales acerca del proyecto"""

	nombre: str = Field(
		description="Nombre del proyecto"
	)
	descripcion: str = Field(
		description="Descripción corta del propósito y objetivos del proyecto"
	)
	tipo_proyecto: Literal["aplicación web", "aplicación móvil", "herramienta CLI", "otro"] = Field(
		description="Tipo principal de proyecto",
		default="aplicación web"
	)
	usuarios_demograficos: Optional[str] = Field(
		description="Objetivo demográfico del proyecto",
		default=None
	)
	funcionalidades_principales: list[str] = Field(
		description="Funcionalidades clave o metas generales",
		default_factory=list
	)


class StackTecnologico(BaseModel):
	"""Definición del stack tecnológico seleccionado para el proyecto"""

	frontend: Optional[str] = Field(
		description="Framework o librería principal de frontend (ej: React, Vue)",
		default=None
	)
	backend: Optional[str] = Field(
		description="Framework o entorno de backend (ej: Node.js, Django, Spring)",
		default=None
	)
	base_datos: Optional[str] = Field(
		description="Sistema de gestión de base de datos (ej: PostgreSQL, Firebase, MongoDB)",
		default=None
	)
	estilo_api: Optional[Literal["REST", "GraphQL", "ninguna", "no estoy seguro"]] = Field(
		description="Estilo de API elegido para exponer funcionalidades",
		default="no estoy seguro"
	)
	alojamiento: Optional[str] = Field(
		description="Plataforma de despliegue o alojamiento (ej: Vercel, Heroku, AWS)",
		default=None
	)


class Tarea(BaseModel):
	titulo: str = Field(
		description="Nombre corto y descriptivo para identificar la tarea")
	descripcion: str = Field(
		description="Descripción de la tarea que se debe llevar a cabo")
	interesado: Optional[Literal[
		"usuario", "desarrollador", "Administrador"
	]] = Field(
		description="El tipo de persona interesada en que esta tarea se realice",
		default=None
	)
	purpose: str = Field(description="Por qué se quiere realizar esta tarea")
	tiempo_estimado: str = Field(
		description="Aproximación del tiempo que el desarrollaodo debería tomarse para realizar la tarea")


class EstadoProgreso(BaseModel):
	"""Estado actual del progreso del proyecto"""

	etapa_actual: Literal[
		"idea", "definiendo alcance", "diseñando", "programando", "depurando", "probando", "desplegando"
	] = Field(
		description="Etapa actual en el ciclo de desarrollo",
		default="idea"
	)
	tareas_completadas: list[Optional[Tarea]] = Field(
		description="Lista de objetivos completados (ej: 'crear repositorio', 'MVP funcional')",
		default_factory=list
	)
	tareas_por_completar: list[Optional[Tarea]] = Field(
		description="Lista de pasos que aun no se han completado",
		default_factory=list
	)


class ContextoProyecto(BaseModel):
	"""Contexto completo del proyecto, incluyendo desarrollador, metadatos, stack y progreso"""

	desarrollador: PerfilDesarrollador = Field(
		description="Perfil del desarrollador a cargo del proyecto"
	)
	proyecto: MetadatosProyecto = Field(
		description="Metadatos generales y descripción del proyecto"
	)
	stack_tecnologico: StackTecnologico = Field(
		description="Stack tecnológico elegido para el proyecto"
	)
	progreso: EstadoProgreso = Field(
		description="Estado actual de avance del proyecto"
	)

## Initialize the model and tools

# Update memory tool
class UpdateMemory(TypedDict):
	""" Decision on what memory type to update """
	update_type: Literal['project', 'todo', 'instructions']

# Initialize the model
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key='sk-proj-t7l5INcC4eUPQaVP2ay4T3BlbkFJJA3LSfWBCZvMM2Bvz4iA')

## Create the Trustcall extractors for updating the project info and ToDo list
info_extractor = create_extractor(
	llm,
	tools=[ContextoProyecto],
	tool_choice="Project",
)

## Prompts 

# Chatbot instruction for choosing what to update and what tools to call 
MODEL_SYSTEM_MESSAGE = """{projektor_role} 

Tienes memoria a largo plazo en la que almacenas información que te permite ayudar al usuario con el desarrollo de su proyecto de software

Aqui teines información sobre el desarrollador (puede estar vacía):
<user-info>
{user_info}
</user-info>

Aquí tienes información sobre el proyecto (puede estar vacío)
<project_info>
{project_info}
</project_info>

Aquí tienes información sobre el estado del proyecto (las tareas hechas y por hacer):
<tasks>
{project_status}
</tasks>

Aquí tienes instrucciones especificadas por el usuario sobre cómo actualizar la lista de tareas
<instructions>
{instructions}
</instructions>

Ejecuta las instrucciones que te dio el usuario de acuerdo a las siguientes reglas:
1. Analiza cuidadosamente el mensaje del usuaio. 
2. Decide si debes o no actualizar tu memoria a largo plazo:
	- si el usuario proporciona información sobre sus habilidades o conocimientos en programación, utiliza `UpdateMemory` con `user`.
	- si el usuario expresa que han habido cambios sobre los detalles del proyecto, utiliza `UpdateMemory` con `project`.
	- si el usuario quiere actualizar el estado de sus tareas, utiliza `UpdateMemory` con `tasks`.
	- si el usuario expresa preferencias sobre cómo actualizar la lista de tareas, utiliza `UpdateMemory` con `instructions`.
3. Comunica al usuario los cambios que haces a tu memoria.
4. No pidas permiso para hacer cambios sobre la información de usuario, o cambios en las tareas, usa tu buen juicio y ejecuta.
5. Luego de ejecutar una herramienta o no, responde naturalmente al usuario."""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Analiza la siguiente interacción. 

Usa las herramientas a tu disposición para guardar información necesaria del proyecto, el usuario o las tareas. 

Usa herramientas en paralelo para actualizar e incertar simultaneamente.

Hora actual: {system_time}"""

# Instructions for updating the ToDo list
CREATE_INSTRUCTIONS = """Analiza la siguiente interacción.

Con base en la siguiente interacción, actualiza tus instrucciones sobre cómo actualizar la lista de tareas. Interpreta los comentarios del usuario sobre cómo le gusta que añadas/edites los items, etc.

Tus instrucciones son:

<instrucciones_actuales>
{current_instructions}
</instrucciones_actuales>"""

UPDATE_USER_INSTRUCTIONS = """Analiza la siguiente interacción y con base en la ella, actualiza la información en el perfil del usuario.
<perfil_de_usuario_actual>
{user_info}
</perfil_de_usuario_actual>"""

## Node definitions

def projektor(state: MessagesState, config: RunnableConfig, store: BaseStore):

	"""Load memories from the store and use them to personalize the chatbot's response."""
	
	# Get the user ID from the config
	configurable = projektor_conf.Configuration.from_runnable_config(config)
	user_id = configurable.user_id
	tasks_category = configurable.tasks_category
	projektor_role = configurable.projektor_role

   # Retrieve project info memories from the store
	namespace = ("project", tasks_category, user_id)
	memories = store.search(namespace)
	project_info = memories[0].value if memories else None
	
	# Retrieve tasks memory from the store
	namespace = ("tasks", tasks_category, user_id)
	memories = store.search(namespace)
	project_status = memories[0].value if memories else None

	# Retrieve custom instructions
	namespace = ("instructions", tasks_category, user_id)
	memories = store.search(namespace)
	instructions = memories[0].value if memories else ""
	
	# Retrieve user profile
	namespace = ("user", tasks_category, user_id)
	memories = store.search(namespace)
	user_profile = memories[0].value if memories else ""
	
	system_msg = MODEL_SYSTEM_MESSAGE.format(
		projektor_role=projektor_role, 
		project_info=project_info, 
		project_status=project_status, 
		instructions=instructions,
		user_info=user_profile)

	# Respond using memory as well as the chat history
	response = llm.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)]+state["messages"])

	return {"messages": [response]}

def update_user(state: MessagesState, config: RunnableConfig, store: BaseStore):
	"""Load memories from the store and use them to personalize the chatbot's response."""
	
	# Get the user ID from the config
	configurable = projektor_conf.Configuration.from_runnable_config(config)
	user_id = configurable.user_id
	tasks_category = configurable.tasks_category

   # Retrieve user profile from the store
	namespace = ("user", tasks_category, user_id)

	user_info = store.search(namespace, "user_profile")


	system_msg = UPDATE_USER_INSTRUCTIONS.format(user_info=user_info)
	
	new_memory = llm.invoke(
		[SystemMessage(content=system_msg)]
		+ state['messages'][:-1]
		+ [HumanMessage(content="Por favor actualiza mi perfil de usuario, con base en esta información.")])

	store.put(namespace, "user_profile", {"memory": new_memory.content})
	tool_calls = state['messages'][-1].tool_calls
	# Return tool message with update verification
	return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id":tool_calls[0]['id']}]}



def update_project(state: MessagesState, config: RunnableConfig, store: BaseStore):

	"""Reflect on the chat history and update the memory collection."""
	
	# Get the user ID from the config
	configurable = projektor_conf.Configuration.from_runnable_config(config)
	user_id = configurable.user_id
	tasks_category = configurable.tasks_category

	# Define the namespace for the memories
	namespace = ("project", tasks_category, user_id)

	# Retrieve the most recent memories for context
	existing_items = store.search(namespace)

	# Format the existing memories for the Trustcall extractor
	tool_name = "ContextoProyecto"
	existing_memories = ([
			(existing_item.key, tool_name, existing_item.value) 
			for existing_item in existing_items] if existing_items else None)

	# Merge the chat history and the instruction
	TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(system_time=datetime.now().isoformat())
	updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

	# Invoke the extractor
	result = info_extractor.invoke({"messages": updated_messages, "existing": existing_memories})

	# Save save the memories from Trustcall to the store
	for r, rmeta in zip(result["responses"], result["response_metadata"]):
		store.put(namespace,
			rmeta.get("json_doc_id", str(uuid.uuid4())),
			r.model_dump(mode="json"))
	tool_calls = state['messages'][-1].tool_calls

	# Return tool message with update verification
	return {"messages": [{"role": "tool", "content": "updated project", "tool_call_id":tool_calls[0]['id']}]}

def update_tasks(state: MessagesState, config: RunnableConfig, store: BaseStore):

	"""Reflect on the chat history and update the memory collection."""
	
	# Get the user ID from the config
	configurable = projektor_conf.Configuration.from_runnable_config(config)
	user_id = configurable.user_id
	tasks_category = configurable.tasks_category

	# Define the namespace for the memories
	namespace = ("tasks", tasks_category, user_id)

	# Retrieve the most recent memories for context
	existing_items = store.search(namespace)

	# Format the existing memories for the Trustcall extractor
	tool_name = "Tarea" # spanish, since it's the name of the class
	existing_memories = (
		[(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items]
		if existing_items
		else None)

	# Merge the chat history and the instruction
	TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(system_time=datetime.now().isoformat())
	updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

	# Initialize the spy for visibility into the tool calls made by Trustcall
	spy = Spy()
	
	# Create the Trustcall extractor for updating the ToDo list 
	todo_extractor = create_extractor(
		llm,
		tools=[Tarea],
		tool_choice=tool_name,
		enable_inserts=True
	).with_listeners(on_end=spy)

	# Invoke the extractor
	result = todo_extractor.invoke({"messages": updated_messages, 
									"existing": existing_memories})

	# Save save the memories from Trustcall to the store
	for r, rmeta in zip(result["responses"], result["response_metadata"]):
		store.put(namespace,
				  rmeta.get("json_doc_id", str(uuid.uuid4())),
				  r.model_dump(mode="json"),
			)
		
	# Respond to the tool call made in projektor, confirming the update    
	tool_calls = state['messages'][-1].tool_calls

	# Extract the changes made by Trustcall and add the the ToolMessage returned to projektor
	todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
	return {"messages": [{"role": "tool", "content": todo_update_msg, "tool_call_id":tool_calls[0]['id']}]}

def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):

	"""Reflect on the chat history and update the memory collection."""
	
	# Get the user ID from the config
	configurable = projektor_conf.Configuration.from_runnable_config(config)
	user_id = configurable.user_id
	tasks_category = configurable.tasks_category
	
	namespace = ("instructions", tasks_category, user_id)

	existing_memory = store.get(namespace, "user_instructions")
		
	# Format the memory in the system prompt
	system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_memory.value if existing_memory else None)
	new_memory = llm.invoke(
		[SystemMessage(content=system_msg)]
		+ state['messages'][:-1] 
		+ [HumanMessage(content="Porfavor actualiza las instrucciones, basándote en esta interacción.")])

	# Overwrite the existing memory in the store
	store.put(namespace, "user_instructions", {"memory": new_memory.content})
	tool_calls = state['messages'][-1].tool_calls
	# Return tool message with update verification
	return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id":tool_calls[0]['id']}]}

# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_tasks", "update_instructions", "update_project", "update_user"]: # type: ignore

	"""Reflect on the memories and chat history to decide whether to update the memory collection."""
	message = state['messages'][-1]
	if len(message.tool_calls) ==0:
		return END
	else:
		tool_call = message.tool_calls[0]
		if tool_call['args']['update_type'] == "project":
			return "update_project"
		elif tool_call['args']['update_type'] == "user":
			return "update_user"
		elif tool_call['args']['update_type'] == "tasks":
			return "update_tasks"
		elif tool_call['args']['update_type'] == "instructions":
			return "update_instructions"
		else:
			raise ValueError

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=projektor_conf.Configuration) # type: ignore

# Define the flow of the memory extraction process
builder.add_node(projektor)
builder.add_node(update_tasks)
builder.add_node(update_project)
builder.add_node(update_instructions)
builder.add_node(update_user)

# Define the flow 
builder.add_edge(START, "projektor")
builder.add_conditional_edges("projektor", route_message)
builder.add_edge("update_tasks", "projektor")
builder.add_edge("update_project", "projektor")
builder.add_edge("update_instructions", "projektor")
builder.add_edge("update_user", "projektor")

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()
# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()
# Save Memories to RAM
# graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory) 

# need to do this to be able to use the context managers for SqliteSaver and SqliteStore
def compile_graph(checkpointer, store):
	return builder.compile(checkpointer=checkpointer, store=store)
