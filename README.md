# Bias in Large Language Models (LLMs) Across Languages

This repository contains code and resources for investigating bias in Large Language Models (LLMs) across multiple languages. The project aims to analyze and mitigate biases present in LLMs in medical text classification across multiple languages.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Data](#data)
- [Analysis](#analysis)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Bias in Large Language Models (LLMs) Across Languages is a research project dedicated to studying and addressing biases that arise in text generation by LLMs when dealing with different languages. This research explores how LLMs may produce biased or stereotypical content in multiple languages and seeks to develop methods to reduce such biases.

## Setup

### Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.9.18
- Required Python packages (specified in `requirements.txt`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/dsrestrepo/MIT_LLMs_Language_bias.git
   cd MIT_LLMs_Language_bias
    ```


2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```


4. Set up your OpenAI API key (Not required for Llama models):

Create a `.env` file in the root directory.

Add your OpenAI API key to the `.env` file:

```makefile
OPENAI_API_KEY=your_api_key_here
```

Make sure you have a valid OpenAI API key to access the language model.

## Data
This project uses a dataset with medical tests in different languages. Place the required dataset in the `data/` directory.

## Usage

Run main.py from the command line with the desired options. Here's an example command:

```bash
python main.py --csv_file data/YourMedicalTestQuestions.csv --model gpt-3.5-turbo --temperature 0.5 --n_repetitions 3 --reasoning --languages english portuguese french
```

The script accepts the following arguments:

- --csv_file: Specify the path to the CSV file containing your medical test questions.
- --model: Choose the LLM to be used gpt-3.5-turbo, gpt-4, Llama-2-7b, Llama-2-13b, or Llama-2-70b.
- --temperature: Set the temperature parameter for text generation (default is 0.0).
- --n_repetitions: Define the number of times each question will be asked to the model. This is useful to measure model's consistency.
- --reasoning: Enable reasoning mode to include explanations for responses. If this argument is not provided, the script will only generate responses. This argument increases the number of tokens used and may result in higher costs.
- --languages: Provide a list of languages to process the questions (space-separated). **The name of the questions should match with the column names containing the questions in the CSV file**.

The script will process the questions, generate responses, and save the results in a CSV file.

Alternatively, you can run the jupyter notebook `main.ipynb` to run the code.


We also provide a more customizable option using the class GPT and Llama. You can import the class and use it to generate responses from the model, change the prompt, and more. See the files `customized_gpt.ipynb` and `customized_llama.ipynb` for examples.


## Analysis
The analysis results, including bias assessment and mitigation strategies, will be documented in the results/ directory. This is where you can find the results of the test in the LLM across languages.

## Contributing
Contributions to this research project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or research.
3. Make your changes.
4. Create tests.
5. Submit a pull request.

We encourage the community to join our efforts to understand and mitigate bias in LLMs across languages.

## License
This project is licensed under the MIT License.

## Contact

For any inquiries or questions regarding this project, please feel free to reach out:

- **Email:** davidres@mit.edu

## Cite

[Seeing Beyond Borders: Evaluating LLMs in Multilingual Ophthalmological Question Answering](https://ieeexplore.ieee.org/abstract/document/10628549):

```
@INPROCEEDINGS{10628549,
  author={Restrepo, David and Nakayama, Luis Filipe and Dychiao, Robyn Gayle and Wu, Chenwei and McCoy, Liam G. and Artiaga, Jose Carlo and Cobanaj, Marisa and Matos, João and Gallifant, Jack and Bitterman, Danielle S. and Ferrer, Vincenz and Aphinyanaphongs, Yindalon and Anthony Celi, Leo},
  booktitle={2024 IEEE 12th International Conference on Healthcare Informatics (ICHI)}, 
  title={Seeing Beyond Borders: Evaluating LLMs in Multilingual Ophthalmological Question Answering}, 
  year={2024},
  volume={},
  number={},
  pages={565-566},
  keywords={Glaucoma;Large language models;Medical services;Documentation;Retina;Question answering (information retrieval);Ophthalmology;Large Language Models;Language Bias;Health Inequalities;LLMs Evaluation;Medical Board Exam},
  doi={10.1109/ICHI61247.2024.00089}}
```

Accepted to AAAI 2025
[Multi-OphthaLingua: A Multilingual Benchmark for Assessing and Debiasing LLM Ophthalmological QA in LMICs](https://arxiv.org/abs/2412.14304):
```
@misc{restrepo2024multiophthalinguamultilingualbenchmarkassessing,
      title={Multi-OphthaLingua: A Multilingual Benchmark for Assessing and Debiasing LLM Ophthalmological QA in LMICs}, 
      author={David Restrepo and Chenwei Wu and Zhengxu Tang and Zitao Shuai and Thao Nguyen Minh Phan and Jun-En Ding and Cong-Tinh Dao and Jack Gallifant and Robyn Gayle Dychiao and Jose Carlo Artiaga and André Hiroshi Bando and Carolina Pelegrini Barbosa Gracitelli and Vincenz Ferrer and Leo Anthony Celi and Danielle Bitterman and Michael G Morley and Luis Filipe Nakayama},
      year={2024},
      eprint={2412.14304},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.14304}, 
}
```

### You can also see:
https://huggingface.co/datasets/AAAIBenchmark/Multi-Opthalingua
