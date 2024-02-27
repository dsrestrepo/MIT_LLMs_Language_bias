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

from langchain.globals import set_verbose

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

### Models:

# Function to validate JSON format
def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def get_completion_from_chain(chain, question, output_parser):
    
    try:
        #response = chain.predict_and_parse(question=question)
        response = chain.run(question=question)
        if is_valid_json:
            response = output_parser.parse(response)
            return response
        else:
            new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=chain)
            response = new_parser.parse(response)
            return response
    except:
        print("except")
        response = get_completion_from_chain(chain, question, output_parser)
        return response
    return response


def get_completion_from_messages(messages, 
                                 model,
                                 output_parser):

    try:
        response = model(messages)
        response = output_parser.parse(response.content)
        if is_valid_json:
            return response
        else:
            new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=model)
            response = new_parser.parse(response)
            return response
    except:
        response = get_completion_from_messages(messages, model=model)
        return response
    



#### Template for the Questions
def generate_prompt(LANGUAGES, REASONING, Responses=['A', 'B', 'C', 'D']):
    
    delimiter = "####"

    languages_text = ", ".join(LANGUAGES)
    
    responses_text = ", ".join(Responses)

    system_message = f"""You are an expert medical assistant.\
You will be provided with medical queries in this languages: {languages_text}. \
Answer the question as best as possible. 
    """
    
    template = system_message + "\n{format_instructions}\n{question}"


    response_schema = ResponseSchema(name="response",
                                     description=f"This is the option of the correct response. Could be only any of these: {responses_text}")

    if REASONING:
        reasoning_schema = ResponseSchema(name="reasoning",
                                    description="This is the reasons for the answer")
        response_schemas = [response_schema, 
                            reasoning_schema]
    else:
        response_schemas = [response_schema]        

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    return prompt, output_parser


def llm_language_evaluation(path='data/Portuguese.csv', model='gpt-3.5-turbo', temperature=0.0, n_repetitions=1, reasoning=False, languages=['english', 'portuguese'], llm_chain=True):
    
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
            
    prompt, output_parser = generate_prompt(LANGUAGES, REASONING)

    for row in range(df.shape[0]):
        print('*'*50)
        print(f'Question {row+1}: ')
        for language in LANGUAGES:
            print(f'Language: {language}')
            question = df[language][row]
            print('Question: ')
            print(question)
            
            if llm_chain:
                chain = LLMChain(llm=llm, prompt=prompt)
            else:
                messages = prompt.format_prompt(question=question)

            for n in range(N_REPETITIONS): 
                print(f'Test #{n}: ')
                if llm_chain:
                    response = get_completion_from_chain(chain, question, output_parser)
                else:
                    response = get_completion_from_messages(messages, llm, output_parser)

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
        df.to_csv(f"responses/{MODEL}_Temperature{str(TEMPERATURE).replace('.', '_')}.csv", index=False)
    else:
        df.to_csv(f"responses/{MODEL}_Temperature{str(TEMPERATURE).replace('.', '_')}_{N_REPETITIONS}Repetitions.csv", index=False)

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