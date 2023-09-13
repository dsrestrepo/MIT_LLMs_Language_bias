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
   git clone https://github.com/dsrestrepo/LLMs_Language_bias.git
   cd LLMs_Language_bias
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

4. Set up your OpenAI API key:

Create a `.env` file in the root directory.

Add your OpenAI API key to the `.env` file:

```makefile
OPENAI_API_KEY=your_api_key_here
```

Make sure you have a valid OpenAI API key to access the language model.

## Data
This project uses a dataset with medical tests in different languages. Place the required dataset in the `data/` directory.

## Usage

Run the code to run the medical text classification experiments:

```bash
python run_experiments.py
```

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


In this updated README, the focus is on bias analysis in LLMs across languages. You can customize it further to include specific details about your data sources, analysis methodologies, and mitigation strategies related to bias in LLMs across different languages.

## Contact

For any inquiries or questions regarding this project, please feel free to reach out:

- **Email:** davidres@mit.edu
