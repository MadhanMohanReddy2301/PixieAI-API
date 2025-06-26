# import os
# from dotenv import load_dotenv
# from semantic_kernel.functions import KernelArguments
# from semantic_kernel import Kernel
# from semantic_kernel.connectors.mcp import MCPSsePlugin
# from semantic_kernel.connectors.ai import FunctionChoiceBehavior
# from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
# from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
# from plugins.code_interpreter import LocalPythonTool
# from plugins.sql_plugin import SqlitePlugin
# from semantic_kernel.contents import ImageContent
#
# # inside get_agent():
#
# from semantic_kernel.services.kernel_services_extension import DEFAULT_SERVICE_NAME
# import weakref
#
# load_dotenv()
# chat_completion = GoogleAIChatCompletion(gemini_model_id="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"),
#                                          service_id=DEFAULT_SERVICE_NAME)
# AGENT_NAME = "Sql_Agent"
#
#
# class SQL_Agent:
#     async def get_agent(self):
#         agent_kernal = Kernel()
#         agent_kernal.add_service(chat_completion)
#         settings = agent_kernal.get_prompt_execution_settings_from_service_id(chat_completion.service_id)
#         settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
#
#         agent_prompt = """ You are PixieAI, a smart assistant for databases and visualizations. You have access to:
#
# • SQL.get_schema(): returns database schema (JSON)
# • SQL.query_select(sql_query: str): executes SELECT and returns rows (JSON list of dicts)
# • plotting_tool.plot_from_code(df: list_of_dicts, code: str): returns an ImageContent with base64-encoded PNG
#
# When the user asks a question, you must:
#
# You have access to:
# • python.execute_code(code: str): runs Python code and returns any generated files (e.g., charts)
# • ...
# If the user requests a chart:
#   1) Fetch data using SQL.query_select
#   2) Use python.execute_code(...) to produce the chart, saving e.g. `plt.savefig("chart.png")`
#   3) Return a JSON: {"text": "...", "image_file": "chart.png", "image_caption": "..."}
#
#
# """
#         agent_kernal.add_plugin(SqlitePlugin(), plugin_name="SQL")
#         # agent_kernal.add_plugin(PlottingTool(), plugin_name="plotting_tool")
#         agent_kernal.add_plugin(LocalPythonTool(), plugin_name="python")
#
#         agent = ChatCompletionAgent(
#             kernel=agent_kernal,
#             name=AGENT_NAME,
#             instructions=agent_prompt,
#             arguments=KernelArguments(settings=settings),
#         )
#
#         return agent
#
#     async def run(self):
#         sql_agent = await self.get_agent()
#         thread: ChatHistoryAgentThread | None = None
#         while True:
#             user_input = input("Enter something:")
#             async for response in sql_agent.invoke(messages=user_input, thread=thread):
#                 print(f"{response.content}")
#                 thread = response.thread
#
#
# if __name__ == "__main__":
#     import asyncio
#
#     asyncio.run(SQL_Agent().run())