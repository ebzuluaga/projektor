import sys
from rich.console import Console
from rich.prompt import Prompt
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.graph.state import CompiledStateGraph
from projektor import compile_graph

console = Console()

def chat(agent: CompiledStateGraph):
	console.print("[bold green]Chat With Projekfor v0.0.2[/bold green] (type 'exit', 'quit', 'bye' or 'goodbye' to termitate the chat)")

	try:
		thread_id = Prompt.ask("Insert thread_id", default="default")
	except (KeyboardInterrupt, EOFError) as e:  # Ctrl+C or Ctrl+D
		console.print("\n[bold red]Exiting...")
		# raise Exception("Keyboard Interrupt", 1)
		raise e

	thread_id = thread_id or 'default'
	
	console.print(f"[dim]Using thread_id: {thread_id}")

	while True:
		try:
			user_input = Prompt.ask("[bold magenta]You[/ bold magenta]")
		except (KeyboardInterrupt, EOFError):  # Ctrl+C or Ctrl+D
			console.print("\n[bold red]Exiting...")
			break

		if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
			break
		if not user_input:
			continue
		with console.status("Thinking..."):
			# Feed the new message into the graph
			state = agent.invoke(input={"messages": [user_input]},
								 config={"configurable": {"thread_id": thread_id}})
			# LangGraph always returns a MessagesState
			last_msg = state["messages"][-1]

		# Handle structured outputs vs plain text
		if hasattr(last_msg, "content"):
			response = last_msg.content
		elif isinstance(last_msg, dict) and "content" in last_msg:
			response = last_msg["content"]
		else:
			response = str(last_msg)

		console.print(f"[bold cyan]Assistant: {response}")


if __name__ == "__main__":

	with SqliteSaver.from_conn_string("data/checkpoints.sqlite") as sqlite_saver, \
			SqliteStore.from_conn_string("data/store.sqlite") as sqlite_store:

		projektor_graph = compile_graph(sqlite_saver, sqlite_store)
		try:
			chat(projektor_graph)
		except Exception as e:
			console.print(f"[bold red]Unexpected error:[/bold red] {e}")
