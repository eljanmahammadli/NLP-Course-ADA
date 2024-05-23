from tqdm import tqdm
import json
import pandas as pd
from datasets import load_dataset
from os import environ
from google.oauth2 import service_account
from google.cloud import translate
from multiprocessing import Pool
from dotenv import load_dotenv

load_dotenv()


PROJECT_ID = environ.get("PROJECT_ID", "")
print(PROJECT_ID)
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"

dataset = load_dataset("GonzaloA/fake_news")
df = dataset["train"].to_pandas()
# flip labels in label column, 1 to 0 and 0 to 1
df["label"] = df["label"].apply(lambda x: 0 if x == 1 else 1)
df["news"] = df["title"] + "\n\n" + df["text"]
original = df[df["label"] == 0].sample(1500, random_state=42)
fake = df[df["label"] == 1].sample(1500, random_state=42)
# concat original and fake news
df = pd.concat([original, fake])
print(df["label"].value_counts())
print(df.head())

path_to_key = (
    "/Users/eljan/Documents/NLP/NLP-Course-ADA/term-paper/service-account-creds.json"
)
credentials = service_account.Credentials.from_service_account_file(path_to_key)
client = translate.TranslationServiceClient(credentials=credentials)


def translate_sentence_googlecloud(
    text: str, source_language_code="en", target_language_code="az"
) -> translate.Translation:
    try:
        response = client.translate_text(
            parent=PARENT,
            contents=[text],
            source_language_code=source_language_code,
            target_language_code=target_language_code,
        )
        return response.translations[0].translated_text
    except Exception as e:
        print(f"Error translating {text}: {e}")
        return None


def main():
    df["az_news"] = None
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        translated_text = translate_sentence_googlecloud(row["news"])
        df.loc[i, "az_news"] = translated_text


if __name__ == "__main__":
    main()
    print(df.head())
    df.to_csv("translated_data.csv", index=False)
