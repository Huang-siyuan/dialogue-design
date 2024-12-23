from typing import List
import torch

def tokenize(text: str, tokenizer, out_len, pad_token_id):
    """
    Кодирует входную строку в числовое представление с помощью
    кодировщика
    """
    encoded = tokenizer.encode(text, add_special_tokens=True)
    
    if len(encoded) > out_len:
        encoded = encoded[:out_len]
    elif len(encoded) < out_len:
        padding_size = out_len - len(encoded)
        pads = [pad_token_id] * padding_size
        encoded = encoded + pads
        
    return encoded


def pad_bio_texts(bio_texts: List[List[int]], pad_to: int, pad_with: int = -1):
    padded = []
    for bio_text in bio_texts:
        diff = pad_to - len(bio_text)
        if diff > 0:
            padded.append(bio_text + [pad_with]*diff)
            continue
        if diff < 0:
            raise ValueError(f"Текст длиной {len(diff)} превышает размер {pad_to}")
        padded.append(bio_text)
    
    return padded

def one_hot_bio_ints(bio_ints: List[List[int]], n_unique_bio_tags: int):
    one_hots = torch.zeros((len(bio_ints), len(bio_ints[0]), n_unique_bio_tags))
    for i, bio_seq in enumerate(bio_ints):
        for b, elem in enumerate(bio_seq):
            if elem == -1:
                continue
            one_hots[i, b, elem] = 1

    return one_hots


def intents_to_one_hot(intents: List[int], n_unique_intents: int):
    one_hot = torch.zeros((len(intents), n_unique_intents))
    one_hot[torch.arange(len(intents)), intents] = 1
    return one_hot
        
