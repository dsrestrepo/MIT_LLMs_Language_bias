""" Evaluate Medical Tests Classification in LLMS """
## Setup
#### Load the API key and libaries.
import json
import pandas as pd
import os
import openai
from dotenv import load_dotenv, find_dotenv
import argparse
import re
import subprocess

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.llms import LlamaCpp

from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer

#from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain.retrievers import PubMedRetriever
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun

#from langchain.globals import set_verbose

#set_verbose(True)

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import OutputFixingParser

from langchain.chains import LLMChain


### Download LLAMA model:
def download_and_rename(url, filename):
    """Downloads a file from the given URL and renames it to the given new file name.

    Args:
        url: The URL of the file to download.
        new_file_name: The new file name for the downloaded file.
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print(f'Downloading the weights of the model: {url} ...')
    subprocess.run(["wget", "-q", "-O", filename, url])
    print(f'Done!')

def download_hugging_face_model(model_version='Llama-2-7b'):
    if model_version not in ['Llama-2-7b', 'Llama-2-13b', 'Llama-2-70b', 'Mistral-7b']:
        raise ValueError("Options for Llama model should be 7b, 13b or 70b, or Mistral-7b")

    MODEL_URL = {
        'Llama-2-7b': 'https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q8_0.gguf', 
        'Llama-2-13b': 'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q8_0.gguf', 
        'Llama-2-70b': 'https://huggingface.co/TheBloke/Llama-2-70B-chat-GGUF/resolve/main/llama-2-70b-chat.Q5_0.gguf',
        'Mistral-7b': 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q8_0.gguf'
    }

    MODEL_URL = MODEL_URL[model_version]

    model_path = f'Models/{model_version}.gguf'

    if os.path.exists(model_path):
        confirmation = input(f"The model file '{model_path}' already exists. Do you want to overwrite it? (yes/no): ").strip().lower()
        if confirmation != 'yes':
            print("Model installation aborted.")
            return model_path

    download_and_rename(MODEL_URL, model_path)

    return model_path


# Function to validate JSON format
def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except:
        return False


def get_completion_from_messages(messages, 
                                 model):

    #try:
    response = model(messages)['output']
    print('response')
    print(response)
    if is_valid_json(response):
        response = json.loads(response)
        return response
    else:
        if '"response": "a"' in response.lower() or '"response":"a"' in response.lower() or ': "a"' in response.lower() or ': "a"' in response.lower() or '"response": a' in response.lower() or '"response":a' in response.lower() or ': a' in response.lower() or ':a' in response.lower():
            response = {'response': 'a'}
        elif '"response": "b"' in response.lower() or '"response":"b"' in response.lower() or ': "b"' in response.lower() or ': "b"' in response.lower() or '"response": b' in response.lower() or '"response":b' in response.lower() or ': b' in response.lower() or ':b' in response.lower():
            response = {'response': 'b'}
        elif '"response": "c"' in response.lower() or '"response":"c"' in response.lower() or ': "c"' in response.lower() or ': "c"' in response.lower() or '"response": c' in response.lower() or '"response":c' in response.lower() or ': c' in response.lower() or ':c' in response.lower():
            response = {'response': 'c'}
        elif '"response": "d"' in response.lower() or '"response":"d"' in response.lower() or ': "d"' in response.lower() or ': "d"' in response.lower() or '"response": d' in response.lower() or '"response":d' in response.lower() or ': d' in response.lower() or ':d' in response.lower():
            response = {'response': 'd'}
        
        return response



def llm_language_evaluation(path='data/Portuguese.csv', model='gpt-3.5-turbo', temperature=0.0, n_repetitions=1, reasoning=False, languages=['english', 'portuguese'], llm_chain=False):
    
    # Load API key if GPT, or Model if LLAMA
    if 'gpt' in model:
        _ = load_dotenv(find_dotenv()) # read local .env file
        openai.api_key  = os.environ['OPENAI_API_KEY']
        llm = OpenAI(temperature=temperature, model_name=model)
        
    elif 'Llama-2' in model or ('Mistral-7b' in model):    
        
        model_path = download_hugging_face_model(model_version=model)
        llm = LlamaCpp(
            model_path=model_path,
            temperature=temperature,
            n_ctx=2048,
            verbose=False,  # VERBOSE
        )
        
    else:
        print('Model should be a GPT, Llama-2, or Mistral-7b model')
        return 0
    
    #### Load the Constants
    PATH = path # 'data/Portuguese.csv'
    MODEL = model # "gpt-3.5-turbo"
    TEMPERATURE = temperature # 0.0
    N_REPETITIONS = n_repetitions # 1
    REASONING = reasoning # False
    LANGUAGES = languages # ['english', 'portuguese']
    
    ##### RAG:
    
    pubmed = PubMedRetriever()
    wikipedia = WikipediaAPIWrapper()
    search = DuckDuckGoSearchRun()
    
    @tool
    def json_format(response: str) -> dict:
        """Given the correct response's letter a, b, c or d; generates the output json. If input is not a, b, c or d, returns an error message."""
        if response in ['a', 'b', 'c', 'd']:
            return {"response": response}
        else:
            return "Error: response should be a, b, c or d."
    
    ##### Tools:
    tools = [
        Tool(
            name = "Pubmed search",
            func=pubmed.run,
            description="useful for when you need to search for a medical topic, treatment or outcome on pubmed"
        ),
        Tool(
            name = "JSON format",
            func=json_format,
            description="Given the correct response's letter a, b, c or d; generates the output json. If input is not a, b, c or d, returns an error message."
        ),
        Tool(
            name='Wikipedia',
            func= wikipedia.run,
            description="Useful for when you need to look up an specific topic, object, or procedure on wikipedia"
        ), 
        Tool(
            name='DuckDuckGo Search',
            func= search.run,
            description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
        )
    ]

    ##### Agent:
    react_agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors="Check your output and make sure it is a JSON file with the key response and value a letter a, b, c, or d. Make sure you can parse that using Python")#AgentType.REACT_DOCSTORE, verbose=True)


    prompt = '''
Answer the following questions as best you can. You have access to the following tools:

Pubmed search: useful for when you need to search for a medical topic, treatment or outcome on pubmed
JSON format: Given the correct response's letter a, b, c or d; generates the output json. If input is not a, b, c or d, returns an error message.
Wikipedia: Useful for when you need to look up an specific topic, object, or procedure on wikipedia
DuckDuckGo Search: Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input.


Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Pubmed search, JSON format, Wikipedia, DuckDuckGo Search]. Don't use the same tool more than 3 times.
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 4 times maximum, then you should answer the question)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. The final answer should be a JSON object with the key "response" and the value being the letter a, b, c or d with the correct answer.

Begin!

Question: {input}
Thought:{agent_scratchpad}
    '''
    
    react_agent.agent.llm_chain.prompt.template = prompt
    

    ##### Experiments:
    if N_REPETITIONS <= 0 or (N_REPETITIONS != int(N_REPETITIONS)):
        print(f'N_REPETITIONS should be a positive integer, not {N_REPETITIONS}')
        print('N_REPETITIONS will be set to 1')
        N_REPETITIONS = 1

    ### Questions from a csv file:
    df = pd.read_csv(PATH)

    ### Evaluate the model in question answering per language:
    responses = {}
    reasoning = {}
    for language in LANGUAGES:
        responses[language] = [[] for n in range(N_REPETITIONS)]

        if REASONING:
            reasoning[language] = [[] for n in range(N_REPETITIONS)]


    for row in range(df.shape[0]):

        print('*'*50)
        print(f'Question {row+1}: ')
        for language in LANGUAGES:
            print(f'Language: {language}')
            question = df[language][row]
            print('Question: ')
            print(question)

            for n in range(N_REPETITIONS): 
                print(f'Test #{n}: ')
                
                response = get_completion_from_messages(question, react_agent)

                print(response)
            
                # Append to the list:
                responses[language][n].append(response['response'])
                if REASONING:
                    reasoning[language][n].append(response['reasoning'])
                
        print('*'*50)

    ### Save the results in a csv file:
    for language in LANGUAGES:
        if N_REPETITIONS == 1:
            df[f'responses_{language}'] = responses[language][0]
            if REASONING:
                df[f'reasoning_{language}'] = reasoning[language][0]
                
        for n in range(N_REPETITIONS):
            df[f'responses_{language}_{n}'] = responses[language][n]
            if REASONING:
                df[f'reasoning_{language}_{n}'] = reasoning[language][n]

    if not os.path.exists('responses'):
        os.makedirs('responses')
    if N_REPETITIONS == 1:
        df.to_csv(f"responses/rag_{MODEL}_Temperature{str(TEMPERATURE).replace('.', '_')}.csv", index=False)
    else:
        df.to_csv(f"responses/rag_{MODEL}_Temperature{str(TEMPERATURE).replace('.', '_')}_{N_REPETITIONS}Repetitions.csv", index=False)

def main():
    # Add argparse code to handle command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate Medical Tests Classification in LLMS")
    parser.add_argument("--csv_file", default="data/Portuguese.csv", help="Path to the CSV file with the questions")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM to use e.g: gpt-3.5-turbo, gpt-4, Llama-2-7b, Llama-2-13b, or Llama-2-70b")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature parameter of the model between 0 and 1. Used to modifiy the model's creativity. 0 is deterministic and 1 is the most creative")
    parser.add_argument("--n_repetitions", type=int, default=1, help="Number of repetitions to run each experiment. Used to measure the model's hallucinations")
    parser.add_argument("--reasoning", action="store_true", default=False, help="Enable reasoning mode. If set to True, the model will be asked to provide a reasoning for its answer. If set to True the model uses more tokens")
    parser.add_argument("--languages", nargs='+', default=['english', 'portuguese'], help="List of languages")
    parser.add_argument("--llm_chain", action="store_true", default=False, help="Enable the use of ")
    
    args = parser.parse_args()


    PATH = args.csv_file
    MODEL = args.model
    TEMPERATURE = args.temperature
    N_REPETITIONS = args.n_repetitions
    REASONING = args.reasoning
    LANGUAGES = args.languages
    llm_chain = args.llm_chain
    
    llm_language_evaluation(path=PATH, model=MODEL, temperature=TEMPERATURE, n_repetitions=N_REPETITIONS, reasoning=REASONING, languages=LANGUAGES, llm_chain=llm_chain)


if __name__ == "__main__":
    main()