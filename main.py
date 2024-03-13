from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description
from langchain.schema import AgentAction, AgentFinish
from typing import Union,List

from callbacks import AgentCallbackHandler

load_dotenv()

@tool
def get_text_length(text:str) -> int:
    """Returns the length of a text by characters"""
    return len(text)

@tool
def generate_text(topic:str) -> str:
    """Write a description on a given topic within 100 words"""
    print(f"write paragraph enter with {topic=}")
    return ChatOpenAI().predict(text=f"write paragraph about {topic}")

def find_tool_by_name(tools:list[Tool],tool_name:str) -> Tool:
    for tool in tools:
        if tool.name == tool.name:
            return tool
    return ValueError(f"Tool wtih name {tool_name} not found")

if __name__ == "__main__":
    print("hello")
    tools = [get_text_length,generate_text]
    templates = """

    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=templates).partial(tools=render_text_description(tools),
                                                                      tool_names=",".join([t.name for t in tools]))
    llm = ChatOpenAI(
        temperature=0, stop=["\nObservation"], callbacks=[AgentCallbackHandler()]
    )
    intermediate_step = []

    agent =( 
        {"input": lambda x: x["input"],
         "agent_scratchpad":lambda x: format_log_to_str(x["agent_scratchpad"])} 
         | prompt 
         | llm 
         | ReActSingleInputOutputParser()
    )

    agent_step = ""
    while not isinstance(agent_step,AgentFinish):
        agent_step: Union[AgentAction,AgentFinish] = agent.invoke(
            {
                "input": "write a paragraph about dog and then count the length of characters?",
                "agent_scratchpad":intermediate_step
            }
        )
        print(agent_step)
        if isinstance(agent_step,AgentAction):
            tool_name = agent_step.tool
            tool_to_use=find_tool_by_name(tools,tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"{observation}")
            intermediate_step.append((agent_step,str(observation)))

    if isinstance(agent_step,AgentFinish):
        print(agent_step.return_values)

   