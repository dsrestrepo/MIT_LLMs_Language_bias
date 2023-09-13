""" Evaluate Medical Tests Classification in LLMS """

## Setup
#### Load the API key and libaries.
import json
import pandas as pd

import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.environ['OPENAI_API_KEY']

#### Load the Constants
PATH = 'data/Portuguese.csv'
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.0
N_REPETITIONS = 1
REASONING = False
LANGUAGES = ['english', 'portuguese']

if N_REPETITIONS <= 0 or (N_REPETITIONS != int(N_REPETITIONS)):
    print(f'N_REPETITIONS should be a positive integer, not {N_REPETITIONS}')
    print('N_REPETITIONS will be set to 1')
    N_REPETITIONS = 1

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
