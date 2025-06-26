import os
from dotenv import load_dotenv
from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.services.kernel_services_extension import DEFAULT_SERVICE_NAME
from plugins.vector_db import FaissSemanticSearchSkill

load_dotenv()
chat_completion = GoogleAIChatCompletion(gemini_model_id="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"),
                                         service_id=DEFAULT_SERVICE_NAME)
AGENT_NAME = "PixieAgent"


class PixieAgent:
    async def get_agent(self):
        agent_kernal = Kernel()
        agent_kernal.add_service(chat_completion)
        settings = agent_kernal.get_prompt_execution_settings_from_service_id(chat_completion.service_id)
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

        agent_prompt = """You are the PixieAI Virtual Assistant. Your job is to help prospects and clients understand how PixieAI’s Generative AI, LLM, and Agentic AI services can solve their business problems. Always consult the “PixieAI Company Knowledge Base” for accurate details on:

                      • Company Overview, Mission & Vision  
                      • Founders & Leadership  
                      • Core Services and Processes  
                      • Industries Served  
                      • Signature Projects, Metrics & Testimonials  
                      • Pricing, Deployment, and Support FAQs  
                      • Contact Information

                    When someone asks a question:

                      1. Reference the relevant section of the Knowledge Base.  
                      2. Give concise, benefit‐focused answers (e.g., “With our natural‐language database agents, non-technical teams see a 90% drop in support tickets within two weeks.”).  
                      3. If asked for next steps, always include the option to “contact madhanreddy@pixieai.in or pavanreddy@pixieai.in” or “visit pixieai.in for a demo.”  
                      4. If you don’t know the answer, offer to escalate or gather more info.

                    Stay friendly, professional, and solution-oriented. Your goal is to convert queries into engaged leads by showcasing PixieAI’s proven impact.
                # Rules:
                    - Shorten the response as much as possible.
                """
        # agent_kernal.add_plugin(FaissSemanticSearchSkill(), plugin_name="company_info_database")
        # agent_kernal.add_plugin(EnhancedPythonPlugin(),plugin_name="PlottingPlugin")

        agent = ChatCompletionAgent(
            kernel=agent_kernal,
            name=AGENT_NAME,
            instructions=agent_prompt,
            arguments=KernelArguments(settings=settings),
        )

        return agent

    async def run(self):
        sql_agent = await self.get_agent()
        thread: ChatHistoryAgentThread | None = None
        while True:
            user_input = input("Enter something:")
            async for response in sql_agent.invoke(messages=user_input, thread=thread):
                print(f"{response.content}")
                thread = response.thread


if __name__ == "__main__":
    import asyncio

    asyncio.run(PixieAgent().run())