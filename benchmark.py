import json
from enum import Enum
import logging
from pathlib import Path
from typing import Optional, List

import faiss
import numpy as np
import pandas as pd
import ranx
import torch
import typer
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

logger = logging.getLogger(__name__)

MR_TYDI_CORPUS = "castorini/mr-tydi-corpus"
MR_TYDI = "castorini/mr-tydi"

class Language(str, Enum):
    arabic = "arabic"
    bengali = "bengali"
    english = "english"
    indonesian = "indonesian"
    finnish = "finnish"
    korean = "korean"
    russian = "russian"
    swahili = "swahili"
    telugu = "telugu"
    thai = "thai"
    japanese = "japanese"
    combined = "combined"


class Split(str, Enum):
    train = "train"
    dev = "dev"
    test = "test"


def combine_title_text(example):
    example["title_text"] = example["title"] + "\n\n" + example["text"]
    example["length"] = len(example["title_text"].split())
    return example


class Tokenization:
    def __init__(self, tokenizer, field, max_length=256):
        self.tokenizer = tokenizer
        self.field = field
        self.max_length = max_length

    def __call__(self, example):
        return self.tokenizer(
            example[self.field], truncation=True, max_length=self.max_length
        )


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings



def embed(model, dataloader):
    embeddings = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            sentence_embeddings = mean_pooling(
                outputs.last_hidden_state, attention_mask
            )
            embeddings.append(sentence_embeddings.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def embed_example_batch(input_ids, attention_mask, collate_fn, model):
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    batch = collate_fn(batch)

    input_ids = batch["input_ids"].to('cuda')
    attention_mask = batch["attention_mask"].to('cuda')
    with torch.inference_mode(), torch.cuda.amp.autocast():
        outputs = model(input_ids, attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(outputs.last_hidden_state, attention_mask)
    return {"embeddings": sentence_embeddings.cpu()}


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_corpus(corpus, tokenizer, model, batch_size):
    logger.info("Combining title and text")
    corpus = corpus.map(combine_title_text)
    corpus = corpus.sort("length", reverse=True)
    logger.info("Tokenizing")
    corpus = corpus.map(Tokenization(tokenizer, "title_text"), batched=True)

    logger.info("Embedding")
    corpus.set_format(type="torch", columns=["input_ids", "attention_mask"])
    corpus = corpus.map(
        embed_example_batch,
        batched=True,
        batch_size=batch_size,
        input_columns=["input_ids", "attention_mask"],
        #keep_in_memory=True,
        fn_kwargs={
            "collate_fn": DataCollatorWithPadding(tokenizer=tokenizer),
            "model": model,
        },
    )
    corpus.reset_format()
    logger.info("Indexing")
    corpus.add_faiss_index(
        column="embeddings",
        metric_type=faiss.METRIC_INNER_PRODUCT,
        string_factory="Flat",
    )
    return corpus


def prepare_queries(queries, tokenizer, model, batch_size):
    queries = queries.map(Tokenization(tokenizer, "query"), batched=True)
    queries = queries.with_format(
        type="torch", columns=["input_ids", "attention_mask"]
    ).map(
        embed_example_batch,
        batched=True,
        batch_size=batch_size,
        input_columns=["input_ids", "attention_mask"],
        #keep_in_memory=True,
        fn_kwargs={
            "collate_fn": DataCollatorWithPadding(tokenizer=tokenizer),
            "model": model,
        },
    )
    return queries


app=    typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    model_name_or_path: str,
    language: Language,
    split: Split = Split.dev,
    batch_size: int = 256,
    k: int = 100,
    device: str = typer.Argument(get_device, help="Device to use"),
    metrics: List[str] = typer.Option(
        ["mrr@100", "recall@100"], help="Metrics to compute"
    ),
    output_path: Optional[Path] = typer.Option(None, help="Output folder"),
):
    if output_path is None:
        output_path = Path(f"results/{model_name_or_path}/{language.value}/{split.value}")

    logger.info(f"Loading {model_name_or_path} using device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).eval().to(device)


    logger.info("Loading corpus")
    corpus = load_dataset(MR_TYDI_CORPUS, language.value, split="train")
    corpus = prepare_corpus(corpus, tokenizer, model, batch_size)

    logger.info("Loading queries")
    queries = load_dataset(MR_TYDI, language.value, split=split.value)
    queries = prepare_queries(queries, tokenizer, model, batch_size)


    logger.info("Searching")
    results = corpus.search_batch("embeddings", queries["embeddings"].numpy(), k=k)

    logger.info("Evaluating - Building Relevance Labels")
    qrels_dict = {
        qid: {pos["docid"]: 1.0 for pos in pos_list}
        for qid, pos_list in zip(queries["query_id"], queries["positive_passages"])
    }
    qrels = ranx.Qrels(qrels_dict)
    logger.info("Evaluating - Building Run Dictionary")
    run_dict = {
        qid: dict(zip(np.array(corpus["docid"])[docidxs], scores))
        for qid, docidxs, scores in zip(
            queries["query_id"], results.total_indices, results.total_scores
        )
    }
    run = ranx.Run(run_dict)
    logger.info("Evaluating - Running Evaluation")
    metrics = ranx.evaluate(qrels, run, ["mrr@100", "recall@100"])
    print(metrics)

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving Index")
    corpus.save_faiss_index("embeddings", output_path / "corpus_embeddings.faiss")
    corpus.drop_index("embeddings")
    logger.info("Saving Corpus")
    corpus.save_to_disk(output_path / "corpus")

    logger.info("Saving Results")
    with open(output_path / "summary.json", "w") as f:
        json.dump(run.mean_scores, f)
        
    df_metrics = pd.DataFrame(run.scores)
    df_metrics.to_csv(output_path / "metrics.csv", index=True)

    logger.info("Saving run")
    run.save(output_path / "run.json", kind="json")
    qrels.save(output_path / "qrels.json", kind="json")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    app()
