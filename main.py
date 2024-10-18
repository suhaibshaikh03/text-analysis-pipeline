from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict, List
from dotenv import load_dotenv
from IPython.display import Image, display 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.prompts import PromptTemplate

import os
load_dotenv()
llm : ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=os.getenv("GEMINI_API_KEY"),temperature = 0)




class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str


def classification_of_text(state : State) -> State:
    # this node classifies the text into one of the categories : news, blog, research or other
    template : str = """
    Classify the following text into one of the categories: News, Blog, Research, or Other:

    {text}

    Ans:
    """
    prompt = PromptTemplate(
    input_variables = ["text"],
    template = template
    )
    message : HumanMessage = HumanMessage(content=prompt.format(text=state['text']))
    classification = llm.invoke([message]).content
    return {"classification" : classification}


def entity_extraction_node(state: State) -> State:
    # This function extracts all the entities for eg.(Person, Organization, Location) from the text '''
    template ="""
    Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.

    Text:{text}

    Entities:
    """
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}


def summarization_node(state: State) -> State:
    # Summarize the text in one short sentence 
    template = """Summarize the following text in one short sentence.
    
    Text:{text}

    Summary:
    """
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}



builder : StateGraph = StateGraph(State)
builder.add_node("classification_of_text",classification_of_text)
builder.add_node("entity_extraction_node",entity_extraction_node)
builder.add_node("summarization_node",summarization_node)

builder.add_edge(START,"classification_of_text")
builder.add_edge("classification_of_text","entity_extraction_node")
builder.add_edge("entity_extraction_node","summarization_node")
builder.add_edge("summarization_node",END)

graph: CompiledStateGraph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png())) # for display of graph in colab notebook

sample_text = """
OpenAI has announced the GPT-4 model, which is a large multimodal model that exhibits human-level performance on various professional benchmarks. It is developed to improve the alignment and safety of AI systems.
additionally, the model is designed to be more efficient and scalable than its predecessor, GPT-3. The GPT-4 model is expected to be released in the coming months and will be available to the public for research and development purposes.
"""

state_input = {"text": sample_text}
result = graph.invoke(state_input)

print("Classification:", result["classification"])
print("\n")
print("Entities:", result["entities"])
print("\n")
print("Summary:", result["summary"])