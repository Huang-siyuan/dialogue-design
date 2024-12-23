from typing import List

import numpy as np
import torch
import torch.ao.quantization
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTextEncoding, AutoTokenizer

from massive_loader import MassiveDatasetWrapper
from massive_utils import (intents_to_one_hot, one_hot_bio_ints, pad_bio_texts,
                           tokenize)

torch.set_float32_matmul_precision("medium")

USE_DATATYPE = torch.float32 # set to torch.float32 for higher precision?
BATCH_SIZE = 5000 # reduce if you run out of memory

#tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
#bert_model = AutoModelForTextEncoding.from_pretrained("DeepPavlov/rubert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
bert_model = AutoModelForTextEncoding.from_pretrained("cointegrated/rubert-tiny2")

MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[MASK]")
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids("[PAD]")
UNK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[UNK]")

from seqeval.metrics import f1_score
from sklearn import metrics

print(f"{MASK_TOKEN_ID=}; {PAD_TOKEN_ID=}; {UNK_TOKEN_ID=}")

import pickle

MAX_TOKEN_LEN_SIZE = 58

m = MassiveDatasetWrapper(split="train")
m_val = MassiveDatasetWrapper(split="test", unique_words=m.unique_words, unique_bio_tags=m.unique_bio_tags, unique_intents=m.unique_intents)
print(f"Val texts = {len(m.texts)}")

n_unique_intents, n_unique_bio_tokens = len(m.unique_intents), len(m.unique_bio_tags)

with open("massive_ds_train_obj.pickle", "wb") as f:
    pickle.dump(m, f)

class BioDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        n_unique_bio_tokens: int,
        texts_bio: List[List[int]],
        n_unique_intents: int,
        intents: List[int],
    ):
        super().__init__()
        self.n_unique_bio_tokens = n_unique_bio_tokens
        self.texts = torch.as_tensor(
            [
                tokenize(text, tokenizer, MAX_TOKEN_LEN_SIZE, PAD_TOKEN_ID)
                for text in texts
            ]
        ).reshape((len(texts), MAX_TOKEN_LEN_SIZE))
        self.attention_mask = torch.as_tensor(
            [
                [1] * (text.count(" ")+1) + [0] * (MAX_TOKEN_LEN_SIZE - text.count(" ") - 1)
                for text in texts
            ]
        )
        self.one_hot_bio = one_hot_bio_ints(
            pad_bio_texts(texts_bio, MAX_TOKEN_LEN_SIZE, pad_with=m.unique_bio_tags.index("o")), n_unique_bio_tokens
        )
        self.one_hot_intents = intents_to_one_hot(intents, n_unique_intents)


        print(self.texts.shape)
        print(self.attention_mask.shape)
        print(self.one_hot_bio.shape)
        print(self.one_hot_intents.shape)

    def __len__(self):
        return self.texts.shape[0]

    def __getitem__(self, idx):
        return (
            self.texts[idx],
            self.attention_mask[idx],
            self.one_hot_bio[idx],
            self.one_hot_intents[idx],
        )


class LinguaModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_lstm_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_lstm_layers = num_lstm_layers
        
        self.embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        self.model = torch.nn.LSTM(embed_dim, embed_dim, num_layers=num_lstm_layers, dropout=0.2, batch_first=True)
        
    def forward(self, texts):
        embeds = self.embeddings(texts)
        return self.model(embeds)


class Model(torch.nn.Module):
    def __init__(
        self,
        bert_model,
        n_unique_bio_tokens: int,
        n_unique_intents: int,
        lstm_cls_hidden_size: int = 32,
    ):
        super().__init__()
        self.n_unique_bio_tokens = n_unique_bio_tokens
        self.n_unique_intents = n_unique_intents
        
        self.bert = bert_model
        for name, param in self.bert.named_parameters():
            print(f"{name=}")
            if (
                "pooler" not in name    # common for tiny/normal
                #and "layer.11" not in name
                #and "layer.10" not in name
                #and "layer.9" not in name
                #and "layer.8" not in name
                #and "11.output" not in name # normal: change for 2.output for tiny
                #and "11.intermediate" not in name # normal: change for 2.intermediate for tiny
                #and "11.attention.output" not in name # normal: change for 11.attention.output
                and "2.output" not in name
                #and "2.intermediate" not in name
                #and "2.attention" not in name
                #and "layer.2" not in name
                #and "layer.1" not in name
                #and "layer.0" not in name
            ):
                param.requires_grad = False
        
        # rubert-tiny-2 has 83828 tokens vocab
        self.own_embed_dim = 16
        self.own_bio_nn = LinguaModel(vocab_size=83828, embed_dim=self.own_embed_dim, num_lstm_layers=2)
        self.own_intent_nn = LinguaModel(vocab_size=83828, embed_dim=self.own_embed_dim, num_lstm_layers=2)
        
        # 768 / 312
        self.intent_cls = torch.nn.Sequential(
            torch.nn.Linear(312 + self.own_embed_dim, n_unique_intents),
            torch.nn.PReLU(),
            torch.nn.LayerNorm(n_unique_intents),
            torch.nn.Linear(n_unique_intents, n_unique_intents),
            torch.nn.PReLU(),
            torch.nn.LayerNorm(n_unique_intents),
            torch.nn.Linear(n_unique_intents, n_unique_intents),
            torch.nn.PReLU(),
            torch.nn.LayerNorm(n_unique_intents),
            torch.nn.Softmax(dim=1)
        )
        self.bio_cls = torch.nn.Sequential(
            torch.nn.Linear(312 + self.own_embed_dim, n_unique_bio_tokens),
            torch.nn.PReLU(),
            torch.nn.LayerNorm(n_unique_bio_tokens),
            torch.nn.Linear(n_unique_bio_tokens, n_unique_bio_tokens),
            torch.nn.PReLU(),
            torch.nn.LayerNorm(n_unique_bio_tokens),
            torch.nn.Linear(n_unique_bio_tokens, n_unique_bio_tokens),
            torch.nn.PReLU(),
            torch.nn.LayerNorm(n_unique_bio_tokens),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        texts, attention_mask, one_hot_bio, one_hot_intents = x
        #print(f"{texts.shape=}; {attention_mask.shape=}; {one_hot_bio.shape=}; {one_hot_intents.shape=}")

        bert_out = self.bert(texts, attention_mask=attention_mask)
        own_bio_out, (_, _) = self.own_bio_nn(texts)
        own_cls_out = self.own_intent_nn(texts)[0][:, -1, :]
        common_bio = torch.concat([bert_out.last_hidden_state, own_bio_out], dim=2)

        intent_cls = self.intent_cls(torch.concat([bert_out.last_hidden_state[:, -1, :], own_cls_out], dim=1))
        bio_cls = torch.concat([self.bio_cls(common_bio[:, i, :]) for i in range(common_bio.shape[1])], dim=1).reshape(one_hot_bio.shape)

        return intent_cls, bio_cls

ds = BioDataset(
    m.texts, n_unique_bio_tokens, m.bio_as_ints, n_unique_intents, m.intents
)
dl = DataLoader(ds, batch_size=BATCH_SIZE)

val_ds = BioDataset(
    m_val.texts, n_unique_bio_tokens, m_val.bio_as_ints, n_unique_intents, m_val.intents
)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = Model(bert_model, n_unique_bio_tokens, n_unique_intents, lstm_cls_hidden_size=32).to(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

model = model.to(USE_DATATYPE)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.000075)

intent_loss_fn = torch.nn.CrossEntropyLoss()
bio_loss_fn = torch.nn.CrossEntropyLoss()

epochs = 2000

INDEX_OF_O = m.unique_bio_tags.index("o")
print(f"{INDEX_OF_O=}")

try:
    for epoch in range(epochs):
        model.train()
        losses_bio = []
        losses_intent = []
        accuracies = []
        accuracies_b = []
        for batch in dl:
            optimizer.zero_grad()
            texts, attention_mask, one_hot_bio, one_hot_intents = batch
            
            texts = texts.to("cuda:0" if torch.cuda.is_available() else "cpu")
            attention_mask = attention_mask.to(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
            one_hot_bio = one_hot_bio.to("cuda:0" if torch.cuda.is_available() else "cpu")
            one_hot_intents = one_hot_intents.to(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
            
            intent_cls, bio_cls = model(
                (texts, attention_mask, one_hot_bio, one_hot_intents)
            )

            #loss_intent = intent_loss_fn(intent_cls, one_hot_intents)
            loss_intent = ((intent_cls - one_hot_intents)**2).mean() + intent_loss_fn(intent_cls, one_hot_intents)
            # Измеряем точность намерений / measure intent accuracy
            argm_p = torch.argmax(intent_cls, dim=-1)
            argm_r = torch.argmax(one_hot_intents, dim=-1)
            accuracy = (argm_p == argm_r).sum() / argm_p.shape[0]
            accuracies.append(accuracy.item())
            
            # same for bio-tags: cross-entropy is the loss to go
            # It can be used simply, or only for words of the original
            # text, ignoring PADs
            # для био тегов то же самое: нужно бы использовать
            # кросс-энтропию. Можно просто, можно штрафовать только
            # в пределах изначального текста, игнорируя PAD
            #loss_bio = bio_loss_fn(bio_cls, one_hot_bio)
            loss_bio = ((bio_cls - one_hot_bio)**2).mean() + bio_loss_fn(bio_cls, one_hot_bio)
            # Accuracy for bio-tags
            # точность для био-тегов
            argm_b_p = torch.argmax(bio_cls, dim=2).reshape(bio_cls.shape[0], -1)
            argm_b_r = torch.argmax(one_hot_bio, dim=2).reshape(bio_cls.shape[0], -1)
            
            accuracy_b = (argm_b_r == argm_b_p).sum() / (argm_b_p.shape[0]*argm_b_p.shape[1])
            
            losses_bio.append(loss_bio.item())
            accuracies_b.append(accuracy_b.item())

            loss = loss_intent + loss_bio
            if loss.requires_grad:
                loss.backward()
            optimizer.step()

            losses_intent.append(loss_intent.item())

        # Validaton loop
        # Валидационный цикл
        if epoch % 10 == 0:
            losses_bio_val = []
            losses_intent_val = []
            accuracies_val = []
            accuracies_b_val = []
            model.eval()
            with torch.no_grad():
                for batch in val_dl:
                    texts, attention_mask, one_hot_bio, one_hot_intents = batch
                    
                    texts = texts.to("cuda:0" if torch.cuda.is_available() else "cpu")
                    attention_mask = attention_mask.to(
                        "cuda:0" if torch.cuda.is_available() else "cpu"
                    )
                    one_hot_bio = one_hot_bio.to("cuda:0" if torch.cuda.is_available() else "cpu")
                    one_hot_intents = one_hot_intents.to(
                        "cuda:0" if torch.cuda.is_available() else "cpu"
                    )

                    intent_cls, bio_cls = model(
                        (texts, attention_mask, one_hot_bio, one_hot_intents)
                    )
                    
                    loss_intent = intent_loss_fn(intent_cls, one_hot_intents)
                    argm_p = torch.argmax(intent_cls, dim=-1)
                    argm_r = torch.argmax(one_hot_intents, dim=-1)
                    accuracy = (argm_p == argm_r).sum() / argm_p.shape[0]
                    accuracies_val.append(accuracy.item())
                    loss_bio_val = (
                        sum(
                            [
                                bio_loss_fn(bio_cls[:, i, :], one_hot_bio[:, i, :])
                                for i in range(one_hot_bio.shape[1])
                            ]
                        )
                        / one_hot_bio.shape[1]
                    )
                    
                    argm_b_p = torch.argmax(bio_cls, dim=2).reshape(bio_cls.shape[0], -1)
                    argm_b_r = torch.argmax(one_hot_bio, dim=2).reshape(bio_cls.shape[0], -1)
                    
                    accuracy_b_val = (argm_b_r == argm_b_p).sum() / (argm_b_p.shape[0]*argm_b_p.shape[1])

                    loss = loss_intent + loss_bio_val
                    losses_bio_val.append(loss_bio_val.item())
                    losses_intent_val.append(loss_intent.item())
                    accuracies_b_val.append(accuracy_b_val.item())
            
            print(
                f"Epoch {epoch}: CrossEntropy-Intent = {np.mean(losses_intent):.3f}; "
                f"Accuracy-Intent = {np.mean(accuracies):.3f} "
                f"CrossEntropy-Intent Val: = {np.mean(losses_intent_val):.3f} "
                f"Accuracy-Intent Val = {np.mean(accuracies_val):.3f} "
                f"CrossEntropy-BIO = {np.mean(losses_bio):.3f} "
                f"CrossEntropy-BIO Val = {np.mean(losses_bio_val):.3f} "
                f"Accuracy BIO = {np.mean(accuracies_b):.3f} "
                f"Accuracy BIO Val = {np.mean(accuracies_b_val):.3f}"
            )
        else:
            print(
                f"Epoch {epoch}: CrossEntropy-Intent = {np.mean(losses_intent):.3f}; "
                f"Accuracy-Intent = {np.mean(accuracies):.3f} "
                f"CrossEntropy-BIO = {np.mean(losses_bio):.3f} "
                f"Accuracy BIO = {np.mean(accuracies_b):.3f} "
            )
except KeyboardInterrupt:
   print("Останавливаю тренировку!")

# Сохраняем модель / save model
torch.save(model.state_dict(), "./trained_model.pth")
print("Model saved!")
model.load_state_dict(torch.load("./trained_model.pth", weights_only=True))
print("Model loaded!")
with open("massive_ds_train_obj.pickle", "rb") as f:
    m = pickle.load(f)
print("Ds loaded")
# = Тестирование и инференс | Testing and Inference =
txt, atm, ohb, ohi = next(iter(DataLoader(val_ds, batch_size=2000, shuffle=False)))
txt = txt.to("cuda:0" if torch.cuda.is_available() else "cpu")
atm = atm.to("cuda:0" if torch.cuda.is_available() else "cpu")
ogb = ohb.to("cuda:0" if torch.cuda.is_available() else "cpu")
ohi = ohi.to("cuda:0" if torch.cuda.is_available() else "cpu")

print("Test data loaded")
model.eval()
print("Model is in eval mode")
with torch.no_grad():
    intent_cls, bio_cls = model((txt, atm, ohb, ohi))
    intent_cls = intent_cls.detach().cpu().float().numpy()
    bio_cls = bio_cls.detach().cpu().float().numpy()

print(f"{intent_cls=}")
intent_cls = np.argmax(intent_cls, axis=1)
print(f"{intent_cls=}")

bio_cls = np.argmax(bio_cls, axis=2)
print(f"{bio_cls=}")

print("="*50)
predicted_intents = []
real_intents = []
seqeval_accs = []
total_correct = 0
for i in range(intent_cls.shape[0]):
    real_intents.append(m_val.unique_intents[m_val.intents[i]])
    predicted_intents.append(m_val.unique_intents[intent_cls[i]])
    
    for t, bio_tag_idx in enumerate(bio_cls[i]):
        if t >= len(m_val.texts_bio[i]):
            # stop if the length of the original text reached, don't include PADs in testing
            # останавлваемся когда достигнем длинны изначального текста, не включаем PADы
            predicted_tags = [m_val.unique_bio_tags[tag_idx] for tag_idx in bio_cls[i][:t]]
            real_tags = m_val.texts_bio[i][:t]
            seqeval_acc = f1_score([real_tags], [predicted_tags])
            seqeval_accs.append(seqeval_acc)
            if all([p == r for p, r in zip(predicted_tags, real_tags)]) and m_val.intents[i] == intent_cls[i]:
                total_correct += 1
            
            print(
                f"Текст: \"{m_val.texts[i]}\" Намерение (факт | прогноз): {real_intents[-1]} | {predicted_intents[-1]}\n"
                f"BIO-разметка (факт | прогноз): {real_tags} | {predicted_tags}\n"
            )
            break
    
print(f"End-to-End Accuracy = {total_correct / intent_cls.shape[0]:.4f}")
print(f"Accuracy intents: {metrics.accuracy_score(real_intents, predicted_intents):.4f}")
print(f"Mean seqeval F1 (BIO): {np.mean(seqeval_accs):.4f}")

print("="*50)


def safe_word_idx(word, words):
    try:
        return words.index(word)
    except:
        return -1

def custom_predict(text, ds: MassiveDatasetWrapper, model: Model):
    """
    Функция для прогноза на произвольном тексте пользователя
    Function to predict for any user text
    """
    tokens = torch.as_tensor(tokenize(text,  tokenizer, text.count(" ") + 1, PAD_TOKEN_ID)).reshape((1, -1)).to("cuda:0" if torch.cuda.is_available() else "cpu")
    one_hot_bio = torch.zeros((1, tokens.shape[1], len(ds.unique_bio_tags))).to("cuda:0" if torch.cuda.is_available() else "cpu")
    attention_mask = torch.ones((one_hot_bio.shape[0], one_hot_bio.shape[1])).to("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        intent_cls, bio_cls = model((tokens, attention_mask, one_hot_bio, None))
        intent_cls = intent_cls.detach().cpu().float().numpy()
        bio_cls = bio_cls.detach().cpu().float().numpy()
    intent_cls = np.argmax(intent_cls, axis=1)
    bio_cls = np.argmax(bio_cls, axis=2)
    
    predicted_intent = ds.unique_intents[intent_cls[0]]
    predicted_tags = [ds.unique_bio_tags[tag_idx] for tag_idx in bio_cls[0]]
    reverse_intent_markup = []
    opened = False
    for tag, word in zip(predicted_tags, text.split(" ")):
        if tag == "o":
            if opened:
                opened = False
                reverse_intent_markup.append(f"{word}] ")
                print(reverse_intent_markup[-1], end=" ")
            else:
                print(f"{word}", end=" ")
        elif "B-" in tag:
            pure_tag = tag.replace("B-", "")
            reverse_intent_markup.append(f"[{pure_tag}: {word}")
            print(reverse_intent_markup[-1], end=" ")
            opened = True
        elif "I-" in tag:
            pure_tag = tag.replace("I-", "")
            reverse_intent_markup.append(f"{word}")
            print(reverse_intent_markup[-1], end=" ")
            
    return predicted_intent, reverse_intent_markup


custom_predict("напиши жалобу на моего мужа в госуслуги", m, model)
