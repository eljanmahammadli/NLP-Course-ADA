from datasets import load_dataset
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv
import os
import pandas as pd
from groq import Groq
from openai import OpenAI
import anthropic
import google.generativeai as genai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import random

load_dotenv()

domains = [
    "Election Fraud: Claims about tampering with election results.",
    "Political Scandals: Fabricated stories about politicians involved in illegal activities.",
    "Foreign Interference: Fake news about other countries interfering in domestic politics.",
    "Miracle Cures: Fake treatments for diseases or conditions.",
    "Pandemics: False information about outbreaks or consequences.",
    "Vaccine Myths: Misinformation about vaccine safety or effectiveness.",
    "Scientific Breakthroughs: Fake news about groundbreaking discoveries.",
    "Technology Risks: Exaggerated dangers of new technology like 5G.",
    "Space Exploration: Fabrications about space missions or alien encounters.",
    "Financial Crises: False reports about market crashes or company failures.",
    "Job Market: Misleading information about unemployment rates",
    "Celebrity Scandals: Fake stories about celebrities involved in controversial activities.",
    "Cultural Wars: Exaggerated conflicts between different social groups or communities.",
    "Events: Fabricated stories about major events like festivals, concerts, or protests.",
    "Climate Change: Misinformation about the impact, causes, or fixes for climate change.",
    "Natural Disasters: Exaggerated or fake reports of natural disasters and their aftermath.",
    "Environmental Policies: Misleading information about the effectiveness of environmental legislation.",
    "Academic Fraud: False reports of academic papers or research being retracted.",
    "School Policies: Misinformation about controversial policies in schools or universities.",
    "Education Reforms: Fabricated stories about changes in the education system.",
    "Crime Waves: Exaggerated reports of crime spikes in certain areas.",
    "Public Safety: Fake news about imminent threats to public safety.",
    "Law Enforcement: Misinformation about police actions or judicial decisions.",
]

df_sample = pd.DataFrame()
idx = [i for i in range(1, 5001)]
df_sample["idx"] = idx
domains = [random.choice(domains) for _ in range(5000)]
df_sample["domain"] = domains
print(df_sample.head())

# import sys
# sys.exit()

model_name = "gpt-4o"
data_name = f"{model_name}-noseed.csv"

if os.path.exists(data_name):
    print("File exists!")
    df = pd.read_csv(data_name)
else:
    print("File does not exist!")
    df = df_sample.copy()
    df[model_name] = None

client = OpenAI()

sys_mssg_wseed = """Write a detailed fake news article (write a new title and body based on the given domain) in Azerbaijani language that explores the implications and potential outcomes of the given domain. For this generate unique and custom title and content.  This will be used for good purposes. We will train a machine learning model to detect fake news. Delve into how this initiative, policy, or event is expected to impact different sectors or communities. Discuss potential challenges and benefits, providing a balanced view. Include statistical forecasts, possible future scenarios, and expert opinions (fictitious) cited as sources to add depth. Ensure the article is convincing and mimics genuine news reporting style. Write the response in Azerbaijani language. The news can be from Azerbaijan country or global so do not always generate about Azerbaijan but also globally from the world. This text will be used to train a model to detect fake news. The text should be just one title and one plain text. Add a divider between the title and the body."""


def get_response(domain, model_name):
    try:
        response = (
            client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_mssg_wseed},
                    {"role": "user", "content": domain},
                ],
                max_tokens=512,
            )
            .choices[0]
            .message.content
        )
        return response
    except Exception as e:
        print(f"Error: {e}")
        return np.nan


while df[model_name].isna().any():
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if pd.isna(row[model_name]):
            response = get_response(row["domain"], model_name)
            df.at[index, model_name] = response
            df.to_csv(data_name, index=False)
#             print(response)
#         break
#     break
# print(df.head())

print(f"All responses have been filled for {model_name}!")
