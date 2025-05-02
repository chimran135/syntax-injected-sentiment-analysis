# A syntax-Injected Approach for Faster and More Accurate Sentiment Analysis

## Description  
This study addresses the computational bottleneck of traditional parsers (e.g. Stanza) by proposing a SEquence Labeling Syntactic Parser (SELSP) to inject syntax into Sentiment Analysis (SA) system. By treating dependency parsing as a sequence labeling problem, we build sentiment analysis system that is lightweight and efficient, while still providing accuracy and explainability through the explicit use of syntax. We intend our approach to be the backbone of a working product of interest for SMEs to use in production.

## Datasets
To train SELSP parser we used [UD EWT English](https://github.com/UniversalDependencies/UD_English-EWT/tree/master) and [UD Spanish AnCora](https://github.com/UniversalDependencies/UD_Spanish-AnCora/tree/master) and for sentiment analysis we use [OpeNER](https://github.com/jerbarnes/semeval22_structured_sentiment) (English and Spanish) and [Rest-Mex 2023](https://sites.google.com/cimat.mx/rest-mex2023 "Dataset can be obtained with the permission of Rest-Mex 2023") datasets. 

## Code Information and Usage Instructions
- **SELSP Parser**:  
  - `parsers/dependency-parser-english.ipynb`: [This notebook conatins the code with step by step instructions to train the parser for English]  
  - `parsers/dependency-parser-spanish.ipynb`: [This notebook conatins the code with step by step instructions to train the parser for Spanish]  
- **Sentiment Analysis**:  
  - `sentiment-analysis/Sentiment_Analyzer_English.ipynb`: [This notebook conatins the code with step by step instructions to perform sentiment analysis on English datasets]  
  - `sentiment-analysis/Sentiment_Analyzer_Spanish.ipynb`: [This notebook conatins the code with step by step instructions to perform sentiment analysis on Spanish datasets]
  - `sentiment-analysis/Sentiment_Analyzer_Spanish.ipynb`: [This notebook conatins the code with step by step instructions to perform sentiment analysis using Vader]    
- **Evaluation**: [Highlight main functionalities]  
  - `python3 sentiment-analysis/eval/Accuracy_Evaluataion.py /path to output file/output.xlsx`: [Use this script to evaluate the sentiment analysis system]  
## Requirements
The `requirements.txt` file contains the dependencies to run this code.<br>
`pip install -r requirements.txt` 

## Acknowledgments
We acknowledge the European Research Council (ERC), which has funded this research under the Horizon Europe research and innovation programme (SALSA, grant agreement No 101100615), SCANNER-UDC (PID2020-113230RB-C21) funded by MICIU/AEI/10.13039/501100011033, LATCHING (PID2023-147129OB-C21) funded by MICIU/AEI/10.13039/501100011033 and ERDF (EU), Ministry for Digital Transformation and Civil Service and “NextGenerationEU” PRTR under grant TSI-100925-2023-1, Xunta de Galicia (ED431C 2024/02), and Galician Research Center “CITIC”, funded by Xunta de Galicia through the collaboration agreement between the Consellería de Cultura, Educación, Formación Profesional e Universidades and the Galician universities for the reinforcement of the research centres of the Galician University System (CIGUS).

## How to cite?
If you use our code in research please cite our reseach artile:<br>
`Imran Muhammad, Olga Kellert, and Carlos Gómez-Rodríguez. "A Syntax-Injected Approach for Faster and More Accurate Sentiment Analysis." arXiv preprint arXiv:2406.15163 (2024).`
