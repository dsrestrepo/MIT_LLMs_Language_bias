## Setup
#### Load the libaries.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from collections import Counter
import scipy.stats as stats


### Get Responses from text
def get_response(response):
    options = {'a)': 'a', 'b)':'b', 'c)': 'c', 'd)':'d'}
    
    for option in options.keys():
        if option in response:
            return options[option]
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
    totals = [df[f'match_{language}'].sum() for language in languages]
    colors = ['lightblue', 'salmon', 'seagreen', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    colors = colors[:len(languages)]
    bars = ax.bar(languages, totals, color=colors)

    # Add labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=12)

    # Add titles and labels
    plt.xlabel('Language')
    plt.ylabel('Total Matches')
    plt.title('Total Correct Answers by Language')

    # Customize the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}/total_matches_{model}.png', bbox_inches='tight')
    plt.show()

def compare_total_matches_by_group(matches_by_test, languages, model, temperature, n_repetitions):
    
    aggregations = {f'match_{language}': 'sum' for language in languages}
    aggregations['Total'] = 'sum'

    matches_by_test_group = matches_by_test.groupby('theme').agg(aggregations).reset_index()

    # Compare English and Portuguese matches by theme
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set the bar width
    bar_width = 0.3

    # Calculate the positions for the bars
    bar_positions = range(len(matches_by_test_group['theme']))
    bar_positions_language = [p + i * bar_width for p in bar_positions for i, _ in enumerate(languages)]


    # Plot the matches per language
    for i, language in enumerate(languages):
        #ax.bar(bar_positions, matches_by_test_group[F'match_{language}'], width=bar_width, label=language)
        ax.bar(
            [p + i * bar_width for p in bar_positions],
            matches_by_test_group[f'match_{language}'],
            width=bar_width,
            label=language
        )

    # Customize the plot
    ax.set_xlabel('Theme')
    ax.set_ylabel('Total Correct Answers Per Language')
    ax.set_title('Total Correct Answers by Theme')
    ax.set_xticks([p + (bar_width * (len(languages) - 1) / 2) for p in bar_positions])
    ax.set_xticklabels(matches_by_test_group['theme'], rotation=45, ha='right')

    # Add a legend
    plt.legend(title='Language', loc='lower right')

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
        totals = [df_theme[f'match_{language}'].sum() for language in languages]
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = colors[:len(languages)]
        
        plt.bar(languages, totals)
        plt.xlabel('Language')
        plt.ylabel('Correct Answers')
        plt.title(f'Correct Answers By Language in Theme: {theme}')
        
        if '/' not in theme:
            plt.savefig(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}/correct_answers_{model}_{theme}.png', bbox_inches='tight')
        else:
            plt.savefig(f'results/results_{model}_Temperature{temperature}_Repetitions{n_repetitions}/correct_answers_{model}_{theme.replace("/", "-")}.png', bbox_inches='tight')
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