import chromadb
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def create_faq_database():
    # Load the dataset from Hugging Face
    ds = load_dataset("deccan-ai/insuranceQA-v2")

    # Combine all splits into a single DataFrame
    df = pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)
    df["combined"] = "Question: " + df["input"] + " \n Answer:  " + df["output"]
    # Inspect
    print(df.shape)
    df.head()

    # Setting up the Chromadb
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    # Create or get the collection
    collection = chroma_client.get_or_create_collection(name="insurance_FAQ_collection")

    # Collection 1 for insurance Q&A Dataset
    df = df.sample(500, random_state=42).reset_index(drop=True)  # For testing, use a smaller subset
    # Add data to collection
    # here the chroma db will use default embeddings (sentence transformers)
    # Split into batches of <= 5000
    batch_size = 100

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]
        collection.add(
            documents=batch_df["combined"].tolist(),
            metadatas=[{"question": q, "answer": a} for q, a in zip(batch_df["input"], batch_df["output"])],
            ids=batch_df.index.astype(str).tolist()
        )

def test_faq_data():
    ## Testing the retrieval
    query = "What does life insurance cover?"
    collection = chroma_client.get_collection(name="insurance_FAQ_collection")
    results = collection.query(
        query_texts=[query],
        n_results=3,
    )

    for i, m in enumerate(results["metadatas"][0]):
        print(f"Result {i+1}:")
        print("Distance:", results["distances"][0][i])
        print("Q:", m["question"])
        print("A:", m["answer"])
        print("-" * 50)
