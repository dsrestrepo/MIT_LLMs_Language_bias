from src.Language_Evaluation import llm_language_evaluation
from src.data_analysis import run_analysis

import argparse


def main():
    # Add argparse code to handle command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate Medical Tests Classification in LLMS")
    parser.add_argument("--csv_file", default="data/full_dataset.csv", help="Path to the CSV file with the questions")
    parser.add_argument("--model", default="gpt-4", help="LLM to use e.g: gpt-3.5-turbo, gpt-4, Llama-2-7b, Llama-2-13b, or Llama-2-70b")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature parameter of the model between 0 and 1. Used to modifiy the model's creativity. 0 is deterministic and 1 is the most creative")
    parser.add_argument("--n_repetitions", type=int, default=0, help="Number of repetitions to run each experiment. Used to measure the model's hallucinations")
    parser.add_argument("--reasoning", action="store_true", default=False, help="Enable reasoning mode. If set to True, the model will be asked to provide a reasoning for its answer. If set to True the model uses more tokens")
    parser.add_argument("--languages", nargs='+', default=['spanish', 'tagalog', 'portuguese', 'english'], help="List of languages")
    args = parser.parse_args()

    PATH = args.csv_file
    MODEL = args.model
    TEMPERATURE = args.temperature
    N_REPETITIONS = args.n_repetitions
    REASONING = args.reasoning
    LANGUAGES = args.languages

    llm_language_evaluation(path=PATH, model=MODEL, temperature=TEMPERATURE, n_repetitions=N_REPETITIONS, reasoning=REASONING, languages=LANGUAGES)

    
    TEMPERATURE = str(TEMPERATURE).replace('.', '_')
    run_analysis(model=MODEL, temperature=TEMPERATURE, n_repetitions=N_REPETITIONS, languages=LANGUAGES)

if __name__ == "__main__":
    main()