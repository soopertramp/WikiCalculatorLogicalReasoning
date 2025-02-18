import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "WikiCalculatorLogicalReasoning"

# Initialize Wikipedia Tool
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Use this tool to search Wikipedia for information on a given topic."
)

# Streamlit UI Setup
st.set_page_config(page_title="Text to Math Problem Solver & Data Search Assistant", page_icon="🧮", layout="wide")
st.title("🧮 Text to Math Problem Solver Using Google Gemma 2")
st.subheader("Solve math problems, perform logical reasoning, and fetch knowledge from Wikipedia!")
st.sidebar.title("Settings")

# API Key input
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
if not api_key:
    st.warning("⚠️ Please enter your API key to continue.")

# Select model
model_name = st.sidebar.selectbox(
    "Select ChatGroq Model:",
    ["Gemma2-9b-it"]
)

# Initialize Math Tool
if api_key:
    math_llm = ChatGroq(model=model_name, groq_api_key=api_key)
    math_chain = LLMMathChain.from_llm(llm=math_llm)
    calculator = Tool(
        name="Calculator",
        func=math_chain.run,
        description="A tool for solving math-related problems. Provide a mathematical expression as input."
    )
else:
    calculator = None

# Logical Reasoning Prompt
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an intelligent agent tasked with solving mathematical and logical questions. 
    Provide a step-by-step solution with a clear explanation in bullet points.

    Question: {question}
    Answer:
    """
)

# Initialize Reasoning Tool
if api_key:
    reasoning_llm = ChatGroq(model=model_name, groq_api_key=api_key)
    chain = LLMChain(llm=reasoning_llm, prompt=prompt_template)
    reasoning_tool = Tool(
        name="Reasoning Tool",
        func=chain.run,
        description="Use this tool for solving logical and reasoning-based questions."
    )
else:
    reasoning_tool = None

# Tool Selection
tools_selected = st.sidebar.multiselect(
    "Select tools:",
    ["Wikipedia", "Calculator", "Reasoning Tool"],
    default=["Wikipedia", "Calculator", "Reasoning Tool"]
)

tools = []
if "Wikipedia" in tools_selected:
    tools.append(wikipedia_tool)
if "Calculator" in tools_selected and calculator:
    tools.append(calculator)
if "Reasoning Tool" in tools_selected and reasoning_tool:
    tools.append(reasoning_tool)

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state["messages"] = []
    st.rerun()

# Credits section at the bottom of the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by [LangChain](https://github.com/langchain-ai/streamlit-agent)**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Display welcome message if chat is empty
if not st.session_state.messages:
    st.write("")

# Handle user input
if user_input := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    if not api_key:
        st.error("❌ API key is required to proceed.", icon="⚠️")
    else:
        try:
            llm = ChatGroq(model=model_name, groq_api_key=api_key, streaming=True)
            assistant_agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                verbose=True  # Enable verbose for debugging
            )

            with st.chat_message("assistant"), st.spinner("🔍 Generating response..."):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                response = assistant_agent.run(conversation_history, callbacks=[st_cb])

                st.session_state.messages.append({'role': 'assistant', "content": response})
                st.write(response)
        except Exception as e:
            st.error(f"❌ An error occurred: {e}", icon="🚨")