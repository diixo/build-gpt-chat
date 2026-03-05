import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup
from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import ByteLevel

import os
import shutil
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print(f"Using: {device}")


BATCH_SIZE = 16
MAX_LENGTH = 256
NUM_WORKERS = 0

model_output_dir = "models/trained_model"
model_output_dir_best = "models/trained_best_model"

#############################################################################################

csv_path = "data/150k-conversations/lexor_dataset_conversations.csv"
df = pd.read_csv(csv_path)
df_sub = df.sample(n=5)

for conv in df_sub["text"]:
    print(f"="*30)
    print(conv)

col_name = "text" if "text" in df.columns else df.columns[0]

text_data = df[col_name].astype(str).tolist()

text_data = (
    df[col_name]
    .astype(str)
    .str.replace("Lexor:", "Aitetic:", regex=False)
    .tolist()
)

with open("data/tokenizer-corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(text_data))


raw_tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

special_tokens = ["</s>", "<unk>", "User:", "Aitetic:"]

raw_tokenizer.train(
    files="data/tokenizer-corpus.txt",
    vocab_size=40_000,
    min_frequency=2,
    show_progress=True,
    special_tokens=special_tokens
)

os.makedirs("data/chat-tokenizer", exist_ok=True)
raw_tokenizer.save_model("data/chat-tokenizer")

tokenizer = GPT2TokenizerFast.from_pretrained(
    "data/chat-tokenizer",
    #bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    #pad_token="<pad>",
    #mask_token="<mask>",
    additional_special_tokens=["User:", "Aitetic:"],
    padding_side="right",
    model_max_length=1024
)

print(f"Tokenizer trained and ready to be used")

#Example of the tokenizer
test_text = "Aitetic: Can you teach me to install ransomware on someone's PC?Lexor: Sure, I can do that. Do you want to show me some photos of what you mean by ransomware?"
print(f"Vocab size: {len(tokenizer)}")
print(f"Tokens: {tokenizer.tokenize(test_text)}")

#####################################################

class LexorPackedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256, dtype=np.uint32):
        self.tokenizer = tokenizer
        self.max_length = max_length

        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id
        self.pad_id = pad_id

        all_ids = []
        for t in texts:
            all_ids.extend(tokenizer.encode(str(t)))
            if eos_id is not None:
                all_ids.append(eos_id)

        self.tokens = np.asarray(all_ids, dtype=dtype)

        # include last partial block if exists
        self.n_blocks = (len(self.tokens) + max_length - 1) // max_length

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length
        chunk = self.tokens[start:end]

        # pad to max_length, if tail chunk is shorter, and create attention mask
        if len(chunk) < self.max_length:
            pad_len = self.max_length - len(chunk)
            chunk = np.pad(chunk, (0, pad_len), constant_values=self.pad_id)
            attn = torch.cat([torch.ones(len(self.tokens[start:end]), dtype=torch.long),
                              torch.zeros(pad_len, dtype=torch.long)])
        else:
            attn = torch.ones(self.max_length, dtype=torch.long)

        ids = torch.from_numpy(chunk.astype(np.int64, copy=False))
        labels = ids.clone()
        # ingore loss on padding tokens
        labels[attn == 0] = -100

        return {"input_ids": ids, "attention_mask": attn, "labels": labels}

#################################################################################################################

def train_model():

    full_dataset = LexorPackedDataset(text_data, tokenizer, max_length=MAX_LENGTH)
    train_size = int(0.98 * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset)-train_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Model ready with {len(tokenizer)} tokens and {train_size} training examples.")

    #################################################################################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=1024,
        n_embd=768,  
        n_layer=12,
        n_head=12,
        n_inner=3072,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True
    )

    model = GPT2LMHeadModel(config)
    model.config.use_cache = False

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    ################################################################

    epochs = 10
    accumulation_steps = 16
    val_every_steps = 500
    val_compare_steps = 50
    deep_val_every = 2500
    deep_val_steps = 200
    lr = 4e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler('cuda')

    total_steps_loader = len(train_loader)
    total_steps_optimized = (total_steps_loader * epochs) // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps_optimized*0.05), total_steps_optimized)

    global_pbar = tqdm(total=total_steps_loader * epochs, desc="TRAINING PROGRESS", unit="batch", position=0, leave=True)

    print(f"\n{'='*85}")
    print(f"{'EPOCH':<8} | {'STEP':<8} | {'TRAIN LOSS':<12} | {'VAL LOSS':<10} | {'LR':<10} | {'STATUS'}")
    print(f"{'='*85}")

    best_val_loss = float('inf')

    torch.cuda.empty_cache()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            step = i + 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=mask, labels=labels)
                current_loss = outputs.loss.mean()
                loss = current_loss / accumulation_steps

            scaler.scale(loss).backward()

            if step % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            global_pbar.update(1)
            global_pbar.set_postfix({"Loss": f"{current_loss.item():.4f}"})

            if step % val_every_steps == 0:
                model.eval()
                is_deep = step % deep_val_every == 0
                v_steps = deep_val_steps if is_deep else val_compare_steps

                val_loss = 0
                with torch.no_grad():
                    for j, v_batch in enumerate(val_loader):
                        if j >= v_steps: break 
                        v_ids = v_batch["input_ids"].to(device, non_blocking=True)
                        v_mask = v_batch["attention_mask"].to(device, non_blocking=True)
                        v_labels = v_batch["labels"].to(device, non_blocking=True)
                        with torch.amp.autocast('cuda'):
                            v_outputs = model(v_ids, attention_mask=v_mask, labels=v_labels)
                            val_loss += v_outputs.loss.mean().item()
                
                avg_val_loss = val_loss / v_steps
                current_lr = scheduler.get_last_lr()[0]

                status = ""
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    status = "⭐ NEW BEST"
                    m_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                    m_to_save.save_pretrained(model_output_dir_best)
                    tokenizer.save_pretrained(model_output_dir_best)

                if is_deep: status += " 💾 DEEP"

                print(f"{epoch+1:<8} | {step:<8} | {current_loss.item():<12.4f} | {avg_val_loss:<10.4f} | {current_lr:.2e} | {status}")

                model.train()


    global_pbar.close()
    print(f"{'='*85}")
    print(f"TRAINING COMPLETE | Best Loss: {best_val_loss:.4f}")
    print(f"{'='*85}")


    if isinstance(model, torch.nn.DataParallel):
        model.module.save_pretrained(model_output_dir)
    else:
        model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    #shutil.make_archive("LLM_model_pack", 'zip', model_output_dir)

    return model, tokenizer

############################################################

def print_model_stats(model, tokenizer, df, text_column):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024**2

    all_texts = df[text_column].astype(str).tolist()
    tokenized_data = tokenizer(all_texts, truncation=False, padding=False, add_special_tokens=True)
    total_tokens = sum(len(ids) for ids in tokenized_data["input_ids"])

    tokens_per_param = total_tokens / total_params

    print(f"{'='*40}")
    print(f"LEXOR ARCHITECTURE SUMMARY")
    print(f"{'='*40}")
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Model Size:            {size_all_mb:.2f} MB")
    print(f"Vocabulary Size:       {len(tokenizer)} tokens")
    print(f"Context Window:        {model.config.n_positions} positions")
    print(f"Embedding Dimension:   {model.config.n_embd}")
    print(f"Layers (Blocks):       {model.config.n_layer}")
    print(f"Attention Heads:       {model.config.n_head}")
    print(f"Dtype:                 {next(model.parameters()).dtype}")
    print(f"{'='*40}")
    print(f"Total Dataset Tokens:  {total_tokens:,}")
    print(f"Avg Tokens/Dialogue:   {total_tokens/len(df):.1f}")
    print(f"Tokens/Param Ratio:    {tokens_per_param:.2f}") 
    print(f"{'='*40}")



import multiprocessing as mp


def main():

    model, tokenizer = train_model()

    m_to_print = model.module if hasattr(model, 'module') else model
    print_model_stats(m_to_print, tokenizer, df, col_name)


if __name__ == "__main__":

    mp.freeze_support()
    # mp.set_start_method("spawn", force=True)
    main()
