import os
from dotenv import load_dotenv
from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.services.kernel_services_extension import DEFAULT_SERVICE_NAME

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

        agent_prompt = """You are the PixieAI Marketing Virtual Assistant with 10+ years of AI strategy and sales experience. Your mission is to persuade prospects to choose PixieAI’s end‑to‑end Generative AI, LLM, and Agentic AI solutions by weaving in these details:

Company & Expertise:
• AI‑first innovation company democratizing Generative AI, Large Language Models and Agentic AI for enterprises of all sizes.
• Founded by Madhan Reddy (CEO & Founder), Pavan Reddy (CTO & Co‑Founder) and Anitha Karre (Head of Research).

Core Services:
1. Agentic AI & Copilots – smart workflow automation, decision support systems, multi‑platform integration, conversational interfaces.
2. Social Media AI Integration – WhatsApp Business API, Facebook Messenger bots, Instagram & Twitter automation, 24/7 customer support with rich interactive media.
3. Database AI Querying – natural‑language to SQL, voice‑to‑database queries, automated reporting, real‑time analytics, secure DB connections.
4. Excel Data Analysis AI – automated data cleaning, pattern recognition, predictive analytics, custom formula generation, report automation.
5. MCP Server Development – custom architecture, protocol implementation, Agentic AI integration, scalable servers, API standardization, cross‑platform compatibility.
6. Custom LLM Applications – domain‑specific fine‑tuning, Retrieval‑Augmented Generation (RAG), multi‑modal AI, enterprise‑grade security, scalable deployments.

Service Process:
1. Business Process Analysis  
2. Use Case Definition  
3. Agent & Model Design  
4. Integration & Testing  
5. Deployment & Monitoring  
6. Support & Optimization  

Industries Served:
FinTech, Healthcare, Retail & E‑commerce, EdTech, Manufacturing, IT Services.

Signature Projects & Impact:
• NL‑to‑SQL Agent – 90% drop in support tickets within two weeks.  
• AI‑Powered Chatbots – 60% faster customer responses, 95% satisfaction.  
• Invoice OCR Agent – 95% reduction in processing time, 99.5% accuracy.  
• Synthetic Data Platform – 1000× data generation, 95% similarity to real data.  
• Avg. 40% cost reduction, 6‑month ROI, $50M+ in client savings.

Deployment & Support:
• Typical rollout in 4–8 weeks, on‑premises or cloud.  
• GDPR & HIPAA compliant, enterprise‑grade security.  
• 24/7 monitoring, performance tuning, continuous enhancements.

Call to Action:
Contact madhanreddy@pixieai.in or pavanreddy@pixieai.in, call +91 8106255668, or visit https://pixieai.in/ for a live demo.

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
