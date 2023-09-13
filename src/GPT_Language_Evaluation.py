""" Evaluate Medical Tests Classification in LLMS """
## Setup
#### Load the API key and libaries.
import json
import pandas as pd
import os
import openai
from dotenv import load_dotenv, find_dotenv
import argparse

### Model:
def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    
    return response.choices[0].message["content"]

#### Template for the Questions
def generate_question(question, LANGUAGES, REASONING, Responses=['A', 'B', 'C', 'D']):
    
    delimiter = "####"

    languages_text = ", ".join(LANGUAGES)

    if REASONING:
        system_message = f"""
        You will be provided with medical queries in this languages: {languages_text}. \
        The medical query will be delimited with \
        {delimiter} characters.
        Each question will have {len(Responses)} possible answer options.\
        provide the letter with the answer and a short sentence answering why the answer was selected \

        Provide your output in json format with the \
        keys: response and reasoning.

        Responses: {", ".join(Responses)}.

        """
    else:
        system_message = f"""
        You will be provided with medical queries in this languages: {languages_text}. \
        The medical query will be delimited with \
        {delimiter} characters.
        Each question will have {len(Responses)} possible answer options.\
        provide the letter with the answer.

        Provide your output in json format with the \
        key: response.

        Responses: {", ".join(Responses)}.

        """

    user_message = f"""/
    {question}"""
    
    messages =  [  
    {'role':'system', 
     'content': system_message},    
    {'role':'user', 
     'content': f"{delimiter}{user_message}{delimiter}"},  
    ] 
    
    return messages


def gpt_language_evaluation(path='data/Portuguese.csv', model='gpt-3.5-turbo', temperature=0.0, n_repetitions=1, reasoning=False, languages=['english', 'portuguese']):

    #### Load the Constants
    _ = load_dotenv(find_dotenv()) # read local .env file
    openai.api_key  = os.environ['OPENAI_API_KEY']

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

    for row in range(df.shape[0]):
        print('*'*50)
        print(f'Question {row+1}: ')
        for language in LANGUAGES:
            print(f'Language: {language}')
            question = df[language][row]
            print('Question: ')
            print(question)
            messages = generate_question(question, LANGUAGES, REASONING)

            for n in range(N_REPETITIONS): 
                print(f'Test #{n}: ')
                response = get_completion_from_messages(messages, MODEL, TEMPERATURE)
                # Convert the string into a JSON object
                response = json.loads(response)
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
    parser.add_argument("--model", default="gpt-3.5-turbo", help="GPT model to use e.g: gpt-3.5-turbo or gpt-3.4 ")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature parameter of the model between 0 and 1. Used to modifiy the model's creativity. 0 is deterministic and 1 is the most creative")
    parser.add_argument("--n_repetitions", type=int, default=1, help="Number of repetitions to run each experiment. Used to measure the model's hallucinations")
    parser.add_argument("--reasoning", action="store_true", default=False, help="Enable reasoning mode. If set to True, the model will be asked to provide a reasoning for its answer. If set to True the model uses more tokens")
    parser.add_argument("--languages", nargs='+', default=['english', 'portuguese'], help="List of languages")
    args = parser.parse_args()


    PATH = args.csv_file
    MODEL = args.model
    TEMPERATURE = args.temperature
    N_REPETITIONS = args.n_repetitions
    REASONING = args.reasoning
    LANGUAGES = args.languages

    gpt_language_evaluation(path=PATH, model=MODEL, temperature=TEMPERATURE, n_repetitions=N_REPETITIONS, reasoning=REASONING, languages=LANGUAGES)


if __name__ == "__main__":
    main()