
# test_local_gpt2_dialog.py
# pip install -U transformers torch accelerate safetensors

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, GPT2TokenizerFast

device = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_DIR = "models/LLM_model"

assistant = "Lexor"

MAX_LENGTH = 512


def main():

    tokenizer = GPT2TokenizerFast.from_pretrained(
        MODEL_DIR,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["User:", f"{assistant}:"],
        padding_side="right",
        model_max_length=MAX_LENGTH
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    # если pad не задан — ставим его = eos (часто нужно для generate)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"EOS id={tokenizer.eos_token_id}, BOS id={tokenizer.bos_token_id}, PAD id={tokenizer.pad_token_id}")

    history = ""

    gen_cfg = GenerationConfig(
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )

    print("Type 'exit' to stop.\n")
    while True:
        user = input("User: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        #prompt = history + f"User: {user}\n{assistant}:"

        prompt = f"User: {user}\n{assistant}:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen_cfg)

        prompt_len = inputs["input_ids"].shape[1]

        out = out[0][prompt_len:]   # cut out the prompt tokens, keep only the generated part

        full = tokenizer.decode(out, skip_special_tokens=True)

        # cut out the answer from the full generated text — отрезаем всё до последнего "Lexor:", а дальше — до следующего "User:" (если есть)
        tail = full.split(f"{assistant}:")[-1]

        # часто модель тянет дальше "User:" — обрежем
        answer = tail.split("User:")[0].strip()

        print(f"{assistant}: {answer}\n")

        history += f"User: {user}\n{assistant}: {answer}\n"


if __name__ == "__main__":
    main()
