from datasets import load_dataset
from pprint import pprint
from typing import List, Union
import numpy as np


def annot_utt_to_bio(text: str) -> List[str]:
    tokens = text.split(" ")
    bio_tokens = []
    last_token = ""
    token_start_set = False
    for token in tokens:
        if token == ":":
            continue

        if "[" in token:
            pure_token = token.replace("[", "")
            # bio_tokens.append(f"B-{pure_token}")
            last_token = pure_token
            continue
        if last_token != "":
            if not token_start_set:
                bio_tokens.append(f"B-{last_token}")
                token_start_set = True
            else:
                bio_tokens.append(f"I-{last_token}")
            if "]" in token:
                last_token = ""
                token_start_set = False
            continue
        bio_tokens.append("o")

    return bio_tokens


def texts_to_ints(texts: List[str], unique_words: List[str], pad_to=None):
    def safe_find_idx(idx):
        try:
            unique_words.index(idx)
        except KeyError:
            return unique_words.index("<unk>")
        
    def pad_text(ints: List[int]):
        if pad_to is None:
            return ints
        diff = pad_to - len(ints)
        if diff > 0:
            return ints + [unique_words.index("<pad>")]*diff
        if diff < 0:
            raise ValueError(f"Текст длиной {len(diff)} превышает размер {pad_to}")
        return ints
    
    return [
        pad_text([safe_find_idx(word) for word in text.split(" ")])
        for text in texts
    ]
    

def bio_texts_to_ints(texts: List[str], unique_words: List[str], pad_to=None):
    def safe_find_idx(idx):
        try:
            unique_words.index(idx)
        except KeyError:
            return unique_words.index("<unk>")
        
    def pad_text(ints: List[int]):
        if pad_to is None:
            return ints
        diff = pad_to - len(ints)
        if diff > 0:
            return ints + [unique_words.index("<pad>")]*diff
        if diff < 0:
            raise ValueError(f"Текст длиной {len(diff)} превышает размер {pad_to}")
        return ints
    
    return [
        pad_text([safe_find_idx(word) for word in text.split(" ")])
        for text in texts
    ]

class MassiveDatasetWrapper:
    """
    Обертка для Amazon-massive, чтобы работать с данными оттуда в удобном формате
    Wrapper for Amazon-massive, to work with data from there with comfort
    """
    def __init__(
        self,
        split="train",
        unique_words: Union[List[str], None] = None,
        unique_bio_tags: Union[List[int], None]=None,
        unique_intents: Union[List[int], None]=None
):
        """
        split - train / test, loads respective part of massive dataset
            загружает train / test часть датасета massive
        unique_words - список уникальных слов в датасете, передавать
            только для split=test, брать из MassiveDatasetWrapper(split="train").unique_words
            List of unique words in the dataset, pass it only
            for split=test, take it from MassiveDatasetWrapper(split="train").unique_words
        unique_bio_tags - то же самое, только список уникальных тегов BIO-разметки (B-playlist, B-date...)
            брать из DatasetWrapper(split="train").unique_bio_tags
            same, but for unique BIO-markup tags (B-playlist, B-date)
            take it from DatasetWrapper(split="train").unique_bio_tags
        unique_intents: ... --> DatasetWrapper(split="train").unique_intents
        """
        self.split = split

        # Загружаем сам датасет / Loading the dataset
        dataset = load_dataset("AmazonScience/massive", "ru-RU", split=split)

        # Берем оттуда нужные данные и превращаем в обычные питон-списки
        # Take the data, turn it into python lists
        self.texts = list(dataset["utt"])
        self.texts_bio = list(map(annot_utt_to_bio, dataset["annot_utt"]))
        self.intents = list(dataset["intent"])
        
        # Build up unique words
        # собираем уникальные слова
        if unique_words is None:
            self.unique_words = set()
            for text in self.texts:
                for word in text.split(" "):
                    if word not in self.unique_words:
                        self.unique_words.add(word)
            self.unique_words = list(self.unique_words)
            self.unique_words.extend(["<unk>", "<pad>"])
            self.unique_words.sort()
        else:
            self.unique_words = unique_words
        
        # build up unique intents
        # собиарем уникальные намерения
        if unique_intents is None:
            self.unique_intents = np.unique(self.intents)
            self.unique_intents.sort()
        else:
            self.unique_intents = unique_intents
        
        # build up unique bio tags
        # собираем уникальные теги био-разметки
        if unique_bio_tags is None:
            self.unique_bio_tags = set()
            for bio_text in self.texts_bio:
                for bio_token in bio_text:
                    if bio_token not in self.unique_bio_tags:
                        self.unique_bio_tags.add(bio_token)
            self.unique_bio_tags = list(self.unique_bio_tags)
            self.unique_bio_tags.sort()
        else:
            self.unique_bio_tags = unique_bio_tags
        
        # превращаем теги био-разметки в их индексы
        # turn bio-tags into their indecies
        self.bio_as_ints = [
            [self.unique_bio_tags.index(bio_tag) for bio_tag in bio_text]
            for bio_text in self.texts_bio
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.texts_bio[idx]
