## Setup
#### Load the libaries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from collections import Counter
import scipy.stats as stats
import pandas as pd
from scipy.stats import chi2_contingency
import json


# Define a dictionary to map Portuguese themes to English themes
theme_mapping = {
    'anatomia': 'anatomy',
    'córnea': 'cornea',
    'embriologia': 'embryology',
    'farmacologia': 'pharmacology',
    'genética': 'genetics',
    'glaucoma': 'glaucoma',
    'oncologia': 'oncology',
    'refração': 'refraction',
    'retina': 'retina',
    'cirurgia refrativa': 'refractive surgery',
    'cirurgia refrativva': 'refractive surgery',
    'cristalino/catarata': 'lens/cataract',
    'córnea/cristalino': 'cornea/lens',
    'estrabismo': 'strabismus',
    'farmacologia/glaucoma': 'pharmacology/glaucoma',
    'glaucoma/uveíte': 'glaucoma/uveitis',
    'lentes de contato': 'contact lenses',
    'neuroftalmologia': 'neuro-ophthalmology',
    'oncologia/plástica ocular': 'oncology/ocular plastic surgery',
    'plástica ocular': 'ocular plastic surgery',
    'refração/visão subnormal': 'refraction/low vision',
    'retina/oncologia': 'retina/oncology',
    'uveíte': 'uveitis',
    'visão subnormal': 'low vision',
    'óptica': 'optics',
    'óptica/refração': 'optics/refraction'
}


### Get Responses from text
def get_response(response):
    
    # Initial check to see if response is a string and try to parse it as JSON
    if isinstance(response, str):
        try:
            response = response.replace("'", '"')
            response = json.loads(response)  # Attempt to parse the string as JSON
        except json.JSONDecodeError:
            # If JSON parsing fails, it's not a JSON string, so we just proceed with the response as is
            pass

    # If response is now a dictionary (either was initially or successfully parsed from a string)
    if isinstance(response, dict):
        if 'response' in response:
            response = response['response']
        # Look for a direct 'answer' key first
        if 'answer' in response:
            response = response['answer']
        # If 'answer' key is not present, search for any key containing 'answer'
        for key in response:
            if 'answer' in key:
                response = response[key]
                
    if response == {'a': 'falso', 'b': 'falso', 'c': 'falso', 'd': 'verdadero'}:
        return 'd'
    elif response == {'a': 'falso', 'b': 'falso', 'c': 'verdadero', 'd': 'falso'}:
        return 'c'
    elif response == {'a': 'falso', 'b': 'verdadero', 'c': 'falso', 'd': 'falso'}:
        return 'b'
    elif response == {'a': 'verdadero', 'b': 'falso', 'c': 'falso', 'd': 'falso'}:
        return 'a'
    else: 
        response = str(response)
    
    if response.lower() in ['a', 'b', 'c', 'd']:
        return response
    
    options = {'a)': 'a', 'b)':'b', 'c)': 'c', 'd)':'d', 'a,': 'a', 'b,':'b', 'c,': 'c', 'd,':'d', 'a.': 'a', 'b.':'b', 'c.': 'c', 'd.':'d'}
    
    for option in options.keys():
        try:
            if option in response:
                return options[option]
        except:
            print(response)
            return np.nan
    else:
        return response

def clean_responses(df, languages, n_repetitions):

    df['answer'] = df['answer'].str.lower()

    for language in languages:
        if n_repetitions <= 1:
            df[f'responses_{language}'] = df[f'responses_{language}'].str.lower()
            df[f'responses_{language}'] = df[f'responses_{language}'].apply(get_response)
        else:
            for n in range(n_repetitions):
                df[f'responses_{language}_{n}'] = df[f'responses_{language}_{n}'].str.lower()
                df[f'responses_{language}_{n}'] = df[f'responses_{language}_{n}'].apply(get_response)
    
    df['theme'] = df['theme'].str.lower().str.strip()
    # Map the Portuguese themes to English themes for visualization
    df['theme'] = df['theme'].map(theme_mapping)
    
    return df


def get_df(model, temperature, n_repetitions, languages):
    ### Read the csv file with the responses
    if n_repetitions > 1:
        df = pd.read_csv(f"responses/{model}_Temperature{temperature}_{n_repetitions}Repetitions.csv")
    else:
        df = pd.read_csv(f"responses/{model}_Temperature{temperature}.csv")

    # Clean the columns with the responses
    df = clean_responses(df, languages, n_repetitions)
    
    return df


# Function to calculate the most common value and confidence interval for a row
def calculate_most_common_and_ci(row):
    # Count the occurrences of each value
    value_counts = Counter(row)

    # Calculate the mode
    mode = value_counts.most_common(1)[0][0]

    # Calculate the expected frequency of the mode under the null hypothesis
    total_items = len(row)
    expected_frequency = total_items / 4  # Assuming equal probability for each value

    # Perform a chi-squared test to calculate the p-value
    observed_frequency = value_counts[mode]
    chi_squared_statistic = ((observed_frequency - expected_frequency) ** 2) / expected_frequency
    degrees_of_freedom = 4 - 1  # There are 4 possible values, so 4 - 1 degrees of freedom
    p_value = 1 - stats.chi2.cdf(chi_squared_statistic, degrees_of_freedom)

    ratio = observed_frequency / total_items

    return mode, p_value, ratio
 
def get_mode_responses(df, languages, n_repetitions, model, temperature):
    for language in languages:
        cols = [f'responses_{language}_{n}' for n in range(n_repetitions)]
        # Calculate the most common value and confidence interval for each row
        df[f'responses_{language}'], df[f'P-value_{language}'], df[f'ratio_{language}'] = zip(*df[cols].apply(calculate_most_common_and_ci, axis=1))
        
    # Create folder to save the plots and the csv file if it does not exist
    if not os.path.exists(f'results'):
        os.makedirs(f'results')
        
    if not os.path.exists(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}'):
        os.makedirs(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}')
    
    df.to_csv(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}/df_clean_{model}.csv', index=False)
    
    return df

### Data Analysis
def generate_summary_df(df, languages, model, temperature, n_repetitions):

    # Calculate the matches between 'answer' and 'responses'
    for language in languages:
        df[f'match_{language}'] = df['answer'] == df[f'responses_{language}']

    df['Total'] = True

    # Group by 'test', 'year', and 'theme' and calculate the count of matches
    aggregations = {f'match_{language}': 'sum' for language in languages}
    aggregations['Total'] = 'sum'

    matches_by_test = df.groupby(['test', 'year', 'theme']).agg(aggregations).reset_index()

    # Create folder to save the plots and the csv file if it does not exist
    if not os.path.exists(f'results'):
        os.makedirs(f'results')
    if not os.path.exists(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}'):
        os.makedirs(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}')
    matches_by_test.to_csv(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}/matches_results_{model}.csv', index=False)
    
    return matches_by_test

def compare_total_matches(df, languages, model, temperature, n_repetitions):
    # Create a figure with subplots
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the total matches
    totals = [round((df[f'match_{language}'].sum()/df[f'Total'].sum())*100, 1) for language in languages]
    colors = ['lightblue', 'salmon', 'seagreen', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    colors = colors[:len(languages)]
    bars = ax.bar(languages, totals, color=colors)

    # Add labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=16)

    # Add titles and labels
    plt.xlabel('Language', fontsize=18)
    plt.ylabel('Total Matches (%)', fontsize=18)
    plt.title(f'Total Correct Answers by Language(%)', fontsize=20)
    
    # Set the height of the x-values (tick labels) to 16
    ax.set_xticklabels(languages, fontsize=16)

    # Customize the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}/total_matches_{model}.png', bbox_inches='tight')
    plt.show()

def compare_total_matches_by_group(matches_by_test, languages, model, temperature, n_repetitions):
    aggregations = {f'match_{language}': 'sum' for language in languages}
    aggregations['Total'] = 'sum'

    matches_by_test_group = matches_by_test.groupby('theme').agg(aggregations).reset_index()

    # Calculate the percentages for each language
    for language in languages:
        match_column = f'match_{language}'
        matches_by_test_group[f'responses_{language} (%)'] = round((matches_by_test_group[match_column] / matches_by_test_group['Total']) * 100, 2)

    matches_by_test_group.to_csv(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}/matches_by_theme_{model}.csv', index=False)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.2  # Adjust bar width as needed
    index = np.arange(len(matches_by_test_group['theme']))
    offset = bar_width * len(languages) / 2

    for i, language in enumerate(languages):
        values = matches_by_test_group[f'responses_{language} (%)']
        positions = [x + i * bar_width - offset for x in index]
        ax.bar(positions, values, bar_width, label=language)

    ax.set_xlabel('Theme', fontsize=12)
    ax.set_ylabel('Correct Answers (%)', fontsize=12)
    ax.set_title('Correct Answers by Theme for Multiple Languages')
    ax.set_xticks(index)
    ax.set_xticklabels(matches_by_test_group['theme'], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}/matches_by_theme_{model}.png', bbox_inches='tight')
    plt.show()
    
    
    
    
    for theme in matches_by_test['theme']:
        df_theme = matches_by_test[matches_by_test['theme'] == theme].groupby(['theme']).agg(aggregations).reset_index()
        
        # Calculate the ratio of matches as a percentage of the total for each language
        for language in languages:
            df_theme[f'{language}_ratio_percentage'] = (df_theme[f'match_{language}'] / df_theme['Total']) * 100
        
        print(df_theme)

        # Plot the ratio of matches for each theme
        plt.figure(figsize=(6, 6))
        totals = [df_theme[f'{language}_ratio_percentage'].sum() for language in languages]
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = colors[:len(languages)]
        
        plt.bar(languages, totals)
        plt.xlabel('Language')
        plt.ylabel('Correct Answers (%)')
        plt.title(f'Correct Answers (%) By Language in Theme: {theme}')
        
        if '/' not in theme:
            plt.savefig(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}/correct_answers_{model}_{theme}.png', bbox_inches='tight')
        else:
            plt.savefig(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}/correct_answers_{model}_{theme.replace("/", "-")}.png', bbox_inches='tight')
        plt.show()



        
def basic_vs_clinical(matches_by_test, languages, model, temperature, n_repetitions):
    
    # Basic VS Clinical
    test_labels = {
        "Teórica I": "Basic Science",
        "Teórica II": "Clinical/Surgical"
    }

    # Map the 'test' column to their labels
    matches_by_test['test_labels'] = matches_by_test['test'].map(test_labels)

    # Group the data by 'test_labels' and calculate the sum of 'Total' for each group
    aggregations = {f'match_{language}': 'sum' for language in languages}
    aggregations['Total'] = 'sum'

    matches_by_test_group = matches_by_test.groupby('test_labels').agg(aggregations).reset_index()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.15  # Adjust bar width as needed
    index = np.arange(len(matches_by_test_group['test_labels']))
    offset = bar_width * len(languages) / 2

    for i, language in enumerate(languages):
        values = round((matches_by_test_group[f'match_{language}'] / matches_by_test_group['Total']) * 100, 1)
        positions = [x + i * bar_width - offset for x in index]
        bars = ax.bar(positions, values, bar_width, label=language)
        
        # Annotate the values over the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Test Type', fontsize=14)
    ax.set_ylabel('Total Correct Answers (%)', fontsize=14)
    ax.set_title('Total Correct Answers by Test Type (%)', fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(matches_by_test_group['test_labels'], fontsize=12)

    plt.legend(title='Language', loc='upper left', fontsize=10)
    plt.tight_layout()

    plt.savefig(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}/matches_by_type_{model}.png', bbox_inches='tight')
    plt.show()

        

def run_analysis(model, temperature, n_repetitions, languages):
    
    if n_repetitions <= 0 or (n_repetitions != int(n_repetitions)):
        print(f'n_repetitions should be a positive integer, not {n_repetitions}')
        print('n_repetitions will be set to 1')
        n_repetitions = 1

    # Data Preprocessing
    df = get_df(model, temperature, n_repetitions, languages)

    # Data Analysis
    if n_repetitions > 1:
        df = get_mode_responses(df, languages, n_repetitions, model, temperature)
        

    matches_by_test = generate_summary_df(df, languages, model, temperature, n_repetitions)

    # Data Visualization
    compare_total_matches(df, languages, model, temperature, n_repetitions)

    compare_total_matches_by_group(matches_by_test, languages, model, temperature, n_repetitions)
    
    basic_vs_clinical(matches_by_test, languages, model, temperature, n_repetitions)

    
    

def main():
    # Add argparse code to handle command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate Medical Tests Classification in LLMS")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="GPT model to use e.g: gpt-3.5-turbo or gpt-3.4 ")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature parameter of the model between 0 and 1. Used to modifiy the model's creativity. 0 is deterministic and 1 is the most creative")
    parser.add_argument("--n_repetitions", type=int, default=1, help="Number of repetitions to run each experiment. Used to measure the model's hallucinations")
    parser.add_argument("--languages", nargs='+', default=['english', 'portuguese'], help="List of languages")
    args = parser.parse_args()


    MODEL = args.model
    TEMPERATURE = args.temperature
    N_REPETITIONS = args.n_repetitions
    LANGUAGES = args.languages

    TEMPERATURE = str(TEMPERATURE).replace('.', '_')

    run_analysis(model=MODEL, temperature=TEMPERATURE, n_repetitions=N_REPETITIONS, languages=LANGUAGES)


if __name__ == "__main__":
    main()