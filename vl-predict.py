
#!/usr/bin/env python3
"""
SmolVLM-256M (ONNX) minimal vision+text inference demo on CPU.

Features:
- Loads an ONNX VLM session
- Preprocesses image to 512x512 and normalizes to [-1, 1]
- Tokenizes prompt with Hugging Face `tokenizers`
- Runs greedy text generation by repeatedly calling ONNX Runtime

Requirements:
  uv pip install onnxruntime tokenizers huggingface_hub pillow numpy
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image

import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# ------------- CONFIG -------------
# Either set this to a local .onnx file or let the script download from the Hub.
# Example Hub repo for ONNX LLM (SmolLM2). For SmolVLM-256M, use your ONNX export path/repo.
# If you already have your ONNX files locally, set MODEL_ONNX_PATH accordingly.
HF_REPO_ID = os.environ.get("HF_REPO_ID", "HuggingFaceTB/SmolVLM-256M-Instruct-ONNX")  # e.g., "HuggingFaceTB/SmolVLM-256M-Instruct-ONNX"
MODEL_ONNX_PATH = os.environ.get("MODEL_ONNX_PATH", "models/smolvlm-256m.onnx")
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "HuggingFaceTB/SmolLM2-360M")  # tokenizer fallback
IMAGE_PATH = os.environ.get("IMAGE_PATH", "sample.jpg")
PROMPT = os.environ.get("PROMPT", "Describe the image briefly:")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32"))

# If you need attention masks or special tokens:
PAD_TOKEN_ID = int(os.environ.get("PAD_TOKEN_ID", "0"))
EOS_TOKEN_ID = int(os.environ.get("EOS_TOKEN_ID", "2"))  # update per tokenizer
BOS_TOKEN_ID = int(os.environ.get("BOS_TOKEN_ID", "1"))  # update per tokenizer

# ------------- HELPERS -------------

def ensure_model():
    """Ensure ONNX model file exists locally."""
    if os.path.isfile(MODEL_ONNX_PATH):
        return MODEL_ONNX_PATH
    if not HF_REPO_ID:
        raise FileNotFoundError(
            f"ONNX model not found at {MODEL_ONNX_PATH} and HF_REPO_ID env var is empty."
        )
    # Try downloading from the Hub
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=os.path.basename(MODEL_ONNX_PATH))
    return path

def load_tokenizer():
    """
    Load a compatible tokenizer.
    For SmolVLM, many exports use a SmolLM tokenizer or a Llama-like tokenizer.
    Adjust TOKENIZER_ID accordingly (check your model card).
    """
    try:
        tok = Tokenizer.from_pretrained(TOKENIZER_ID)
    except Exception:
        # If hub pull fails, try reading a local JSON
        tok = Tokenizer.from_file("tokenizer.json")
    return tok

def preprocess_image(img_path, size=(512, 512)):
    """Resize to 512x512 and normalize to [-1, 1] (SigLIP-style)."""
    img = Image.open(img_path).convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0      # [0, 1]
    arr = (arr - 0.5) / 0.5                              # [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))                   # HWC -> CHW
    arr = np.expand_dims(arr, 0)                         # NCHW
    return arr

def prepare_text(tok, text):
    enc = tok.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention = np.ones_like(input_ids, dtype=np.int64)
    return input_ids, attention

def greedy_generate(sess, input_names, output_names, image_tensor, input_ids, attention_mask,
                    max_new_tokens=32, eos_id=EOS_TOKEN_ID):
    """
    Minimal greedy decoding loop:
    - Feeds image + current tokens
    - Appends argmax(logits) token each step
    """
    generated = input_ids.copy()
    for _ in range(max_new_tokens):
        feed = {}

        # Map inputs dynamically based on the model's input names
        # Common patterns:
        # - Image: "pixel_values" or "image"
        # - Text: "input_ids", "attention_mask"
        for inp in input_names:
            name = inp.name
            if "pixel" in name or "image" in name:
                feed[name] = image_tensor
            elif "input_ids" in name:
                feed[name] = generated
            elif "attention_mask" in name:
                # build attention mask for current length
                attn = np.ones_like(generated, dtype=np.int64)
                feed[name] = attn
            elif "position_ids" in name:
                # optional; build simple range positions
                seq_len = generated.shape[1]
                pos = np.arange(seq_len, dtype=np.int64)[None, :]
                feed[name] = pos
            else:
                # If the model expects other inputs, try a reasonable default (pad or zeros)
                # You can tailor these based on your model card.
                pass

        outputs = sess.run(None, feed)
        # Find logits among outputs
        # Often named "logits" or last tensor
        logits = None
        for i, out in enumerate(outputs):
            on = output_names[i].name
            if "logits" in on or i == len(outputs) - 1:
                logits = out
        if logits is None:
            raise RuntimeError("Could not locate logits in model outputs.")

        next_id = int(np.argmax(logits[:, -1, :]))  # last-token distribution
        # Append
        generated = np.concatenate([generated, np.array([[next_id]], dtype=np.int64)], axis=1)
        if next_id == eos_id:
            break

    return generated

def decode_ids(tok, ids):
    try:
        return tok.decode(ids[0].tolist())
    except Exception:
        # Fallback simple join
        return " ".join(map(str, ids[0].tolist()))

# ------------- MAIN -------------

def main():
    model_path = ensure_model()
    print(f"[INFO] Using ONNX model: {model_path}")

    # Create session (CPU)
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print("[INFO] Model inputs:")
    for i in inputs:
        print(" -", i.name, i.type, i.shape)
    print("[INFO] Model outputs:")
    for o in outputs:
        print(" -", o.name, o.type, o.shape)

    # Prepare image and text
    image_tensor = preprocess_image(IMAGE_PATH, size=(512, 512)).astype(np.float32)
    tok = load_tokenizer()
    input_ids, attention_mask = prepare_text(tok, PROMPT)

    # Run greedy decode
    t0 = time.time()
    generated = greedy_generate(
        sess,
        input_names=inputs,
        output_names=outputs,
        image_tensor=image_tensor,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS
    )
    dt = (time.time() - t0) * 1000
    print(f"[INFO] Generation latency: {dt:.1f} ms")

    text = decode_ids(tok, generated)
    print("\n--- OUTPUT ---")
    print(text)

if __name__ == "__main__":
    main()