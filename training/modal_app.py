"""
Modal app for training and serving a braille tutor LLM.

Usage:
  modal run modal_app.py::train        # Fine-tune the model
  modal deploy modal_app.py            # Deploy the inference endpoint
  modal run modal_app.py::test_chat    # Test locally
"""

import modal

APP_NAME = "braille-tutor"

app = modal.App(APP_NAME)

# Volumes
hf_cache = modal.Volume.from_name("braille-tutor-hf-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("braille-tutor-adapters", create_if_missing=True)

# Training image
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "trl==0.12.2",
        "datasets>=3.0",
        "bitsandbytes>=0.44",
        "accelerate>=1.0",
        "sentencepiece",
    )
    .add_local_file("braille_tutor_train.jsonl", "/root/braille_tutor_train.jsonl")
)

# Inference image
serve_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "accelerate>=1.0",
        "sentencepiece",
        "fastapi[standard]",
    )
    .add_local_file("ueb_knowledge.json", "/root/ueb_knowledge.json")
)


@app.function(
    image=train_image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    volumes={"/hf-cache": hf_cache, "/adapters": adapter_vol},
)
def train(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    epochs: int = 4,
    lr: float = 2e-4,
    max_seq_length: int = 1024,
):
    """QLoRA fine-tune on braille tutoring data."""
    import json, os
    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf-cache"

    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer
    import torch

    print(f"Training {model_name} for {epochs} epochs on braille tutor data")

    # Load dataset from embedded data
    data_path = "/root/braille_tutor_train.jsonl"
    with open(data_path) as f:
        raw = [json.loads(line) for line in f]
    print(f"Loaded {len(raw)} examples")

    # Format as text for SFTTrainer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir="/hf-cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_example(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )}

    dataset = Dataset.from_list(raw).map(format_example)
    print(f"Formatted {len(dataset)} examples, sample length: {len(dataset[0]['text'])} chars")

    # Load model with QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/hf-cache",
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = "/adapters/braille-tutor-lora"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    adapter_vol.commit()
    hf_cache.commit()

    print(f"Training complete. Adapter saved to {output_dir}")
    return {"status": "done", "output_dir": output_dir, "examples": len(raw)}


# ── Inference endpoint ───────────────────────────────────────────────────
@app.cls(
    image=serve_image,
    gpu="A10G",
    volumes={"/hf-cache": hf_cache, "/adapters": adapter_vol},
    scaledown_window=300,  # Scale to zero after 5 min idle
)
class BrailleTutor:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    @modal.enter()
    def load(self):
        import os, json
        os.environ["HF_HOME"] = "/hf-cache"

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch

        # Load RAG knowledge base
        kb_path = "/root/ueb_knowledge.json"
        if os.path.exists(kb_path):
            with open(kb_path) as f:
                self.kb = json.load(f)
            print(f"Loaded {len(self.kb)} RAG knowledge entries")
        else:
            self.kb = []
            print("WARNING: No knowledge base found")

        adapter_path = "/adapters/braille-tutor-lora"
        print(f"Loading base model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir="/hf-cache"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="/hf-cache",
        )

        if os.path.exists(adapter_path):
            print(f"Loading LoRA adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        else:
            print("WARNING: No adapter found, using base model")

        self.model.eval()
        print("Model loaded and ready")

    def retrieve(self, query: str, top_k: int = 8) -> str:
        """Keyword-based RAG retrieval from the UEB knowledge base."""
        import re
        query_lower = query.lower()
        # Extract meaningful tokens (ignore very short ones)
        tokens = set(re.findall(r'\b[a-z]{2,}\b', query_lower))
        # Also grab single letters if the query asks about them
        single_letters = set(re.findall(r'\bletter\s+([a-z])\b', query_lower))
        single_letters |= set(re.findall(r'\b([a-z])\s+in\s+braille\b', query_lower))
        single_letters |= set(re.findall(r"\bwhat\s+is\s+([a-z])\b", query_lower))

        # Extract dot patterns like "dots 1-2-5" or "1-2-5" for reverse lookup
        dot_patterns = re.findall(r'(?:dots?\s+)?(\d(?:-\d)+)', query_lower)

        # Extract Unicode braille characters for reverse lookup
        braille_chars = [ch for ch in query if '\u2800' <= ch <= '\u28FF']

        scored = []
        for entry in self.kb:
            score = 0
            kw_lower = [k.lower() for k in entry["keywords"]]
            kw_set = set(kw_lower)
            # Exact keyword match
            for kw in kw_lower:
                if kw in query_lower:
                    score += 3
                for token in tokens:
                    if token == kw:
                        score += 2
                    elif token in kw or kw in token:
                        score += 1
            # Single letter matches
            if entry["type"] == "letter" and single_letters:
                for letter in single_letters:
                    if entry["id"] == f"letter_{letter}":
                        score += 10
            # Dot pattern reverse lookup (e.g. "what is dots 1-2-5?")
            for dp in dot_patterns:
                if dp in kw_set or f"dots {dp}" in kw_set:
                    score += 10
            # Unicode braille cell reverse lookup
            for bc in braille_chars:
                if bc in kw_set:
                    score += 10
            # Boost comparisons when "difference" or "vs" in query
            if entry["type"] == "comparison" and any(w in query_lower for w in ["difference", "compare", "vs", "versus"]):
                eid = entry["id"]  # e.g. "compare_d_f"
                parts = eid.split("_")
                if len(parts) == 3 and parts[1] in query_lower and parts[2] in query_lower:
                    score += 10
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        results = scored[:top_k]
        if not results:
            return ""
        facts = "\n".join(f"- {e['fact']}" for _, e in results)
        return facts

    @modal.method()
    def chat(self, messages: list[dict], max_tokens: int = 512) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    @modal.fastapi_endpoint(method="POST")
    def api(self, request: dict) -> dict:
        """HTTP endpoint for the tutor frontend."""
        from fastapi.responses import JSONResponse

        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 512)

        # Extract the user's latest question for RAG retrieval
        user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_query = m.get("content", "")
                break

        # RAG: retrieve relevant UEB facts
        rag_context = self.retrieve(user_query) if user_query else ""

        # Build system prompt with injected facts
        system = (
            "You are a braille tutor. You teach Unified English Braille (UEB) to sighted "
            "and blind learners. You give concise, accurate answers. When showing braille, "
            "use Unicode braille characters (U+2800-U+283F) and always state the dot numbers. "
            "You are not a general-purpose AI. You redirect off-topic questions back to braille. "
            "You are encouraging but honest when a learner makes a mistake."
        )
        if rag_context:
            system += (
                "\n\nREFERENCE FACTS (use these for accuracy -- they are verified UEB data):\n"
                + rag_context
                + "\n\nAlways prefer these reference facts over your own knowledge when answering."
            )

        # Insert or replace system message
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system
        else:
            messages = [{"role": "system", "content": system}] + messages

        response = self.chat.local(messages, max_tokens)
        return JSONResponse(
            content={"response": response},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    @modal.fastapi_endpoint(method="OPTIONS")
    def options(self):
        """Handle CORS preflight."""
        from fastapi.responses import Response
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Max-Age": "86400",
            },
        )


@app.local_entrypoint()
def test_chat():
    """Test the deployed model with a few questions."""
    tutor = BrailleTutor()

    questions = [
        "What is the letter a in braille?",
        "How do numbers work?",
        "What's the difference between d and f?",
        "Teach me Grade 2 contractions",
        "What's the weather like?",
    ]

    system = {
        "role": "system",
        "content": "You are a braille tutor. You teach Unified English Braille (UEB). Give concise, accurate answers with Unicode braille characters and dot numbers.",
    }

    for q in questions:
        print(f"\n{'='*60}")
        print(f"USER: {q}")
        response = tutor.chat.remote([system, {"role": "user", "content": q}])
        print(f"TUTOR: {response}")
