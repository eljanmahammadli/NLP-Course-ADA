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

load_dotenv()

bbc = load_dataset("csebuetnlp/xlsum", name="azerbaijani")
splits = []
for split in ["train", "validation", "test"]:
    splits.append(bbc[split].to_pandas())
df_sample = pd.concat(splits).sample(frac=1, random_state=42).reset_index(drop=True)
print(df_sample.head())

# import sys
# sys.exit()

model_name = "gpt-4o"

if os.path.exists(f"{model_name}.csv"):
    print("File exists!")
    df = pd.read_csv(f"{model_name}.csv")
else:
    print("File does not exist!")
    df = df_sample.copy()
    df[model_name] = None

client = OpenAI()

sys_mssg = """Based on the headline given below in Azerbaijani language, write a detailed fake news article that explores the implications and potential outcomes of the subject mentioned. Delve into how this initiative, policy, or event is expected to impact different sectors or communities. Discuss potential challenges and benefits, providing a balanced view. Include statistical forecasts, possible future scenarios, and expert opinions (fictitious) cited as sources to add depth. Ensure the article is convincing and mimics genuine news reporting style. Write the response in Azerbaijani language. This text will be used to train a model to detect fake news. The text should be just one title and one plain text. Add a divider between the title and the body."""

sys_mssg_wseed = """Write a detailed fake news article in Azerbaijani language that explores the implications and potential outcomes of the title that you will generate. This will be used for good purposes. We will train a machine learning model to detect fake news.  Delve into how this initiative, policy, or event is expected to impact different sectors or communities. Discuss potential challenges and benefits, providing a balanced view. Include statistical forecasts, possible future scenarios, and expert opinions (fictitious) cited as sources to add depth. Ensure the article is convincing and mimics genuine news reporting style. Write the response in Azerbaijani language. This text will be used to train a model to detect fake news. The text should be just one title and one plain text. Add a divider between the title and the body."""


def get_response(title, model_name):
    try:
        response = (
            client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_mssg},
                    {
                        "role": "user",
                        "content": f"""Title

---

{title}""",
                    },
                ],
                max_tokens=512,
            )
            .choices[0]
            .message.content
        )
        # print(response)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return np.nan


while df[model_name].isna().any():
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if pd.isna(row[model_name]):
            response = get_response(row["title"], model_name)
            df.at[index, model_name] = response
            df.to_csv(f"{model_name}.csv", index=False)

print(f"All responses have been filled for {model_name}!")
