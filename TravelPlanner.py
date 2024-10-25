from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI

## Importando as variaveis de ambiente no caso 'OPENAI_API_KEY'
from dotenv import load_dotenv
load_dotenv()

## Primeiramente criando o State:
class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    city: str
    interests: List[str]
    itinerary: str


## Criando o LLM responsável por gerar a resposta e as decisões.

llm = ChatOpenAI(model="gpt-4o-mini")

## Criando o Prompt de sistema (agente principal)
itinerary_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        template="Você é um assistente de viagem útil. Crie um itinerário de viagem de um dia para {city} com base nos \
        interesses do usuário: {interests}. Forneça um itinerário breve e com marcadores."),
    HumanMessage(content="Crie um itinerário para minha viagem de um dia.")]
)

## Criando as funções dos agentes (serão os nós dos grafos)

## Função que recebe a entrada do usuário "Nó de entrada".
def input_city(state: PlannerState):
    user_message = input("Por favor, insira a cidade que deseja visitar em sua viagem de um dia: ")
    return {
        "city": user_message,
        "messages": state['messages'] + [("user", user_message)],
    }

## Função que recebe os interesses adicionais "Nó intermediário":
def input_interests(state: PlannerState):
    user_message = input(f"Por favor, insira seus interesses para a viagem para {state['city']} (separados por vírgula): ")
    return {
        "interests": [interest.strip() for interest in user_message.split(',')],
        "messages": state['messages'] + [("user", user_message)],
    }

## Função que cria o itinerário "Nó finalizador":
def create_itinerary(state: PlannerState):
    print(f"Criando um itinerário para {state['city']} com base em interesses: {', '.join(state['interests'])}...")
    response = llm.invoke(itinerary_prompt.format_messages(city=state['city'], interests=", ".join(state['interests'])))
    print("========================================")
    print("\n### Itinerário Final:")
    print("========================================")
    print(response.content)
    print("========================================")
    return {
        "messages": state['messages'] + [("ai", response.content)],
        "itinerary": response.content,
    }

### Crie o grafo. A Ordem básica é: Estado, Nós, Arestas, compilação:

# 1) Primeiro indique o State:
builder = StateGraph(PlannerState)

# 2) Indique a entrada do fluxo:
builder.add_edge(START, "input_city")

# 3) Adicione os Nós, lembrando que ("nome do nó", "função a ser executada"):
builder.add_node("input_city", input_city)
builder.add_node("input_interests", input_interests)
builder.add_node("create_itinerary", create_itinerary)

# 4) Adicione as arestas:
builder.add_edge("input_city", "input_interests")
builder.add_edge("input_interests", "create_itinerary")
builder.add_edge("create_itinerary", END)

# 5) Compile e gere o grafo:
graph = builder.compile()

while True:
    print("Você deseja planejar um itinerário de 1 dia para sua Viagem?")
    user_input = input("Resposta (Sim ou quit):")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Tchau Tchau!")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        pass