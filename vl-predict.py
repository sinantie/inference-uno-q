from transformers import AutoConfig, AutoProcessor
from transformers.image_utils import load_image
import onnxruntime
import numpy as np

# 1. Load models
## Load config and processor
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
config = AutoConfig.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

## Load sessions
## !wget https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/onnx/vision_encoder.onnx
## !wget https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/onnx/embed_tokens.onnx
## !wget https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/onnx/decoder_model_merged.onnx
vision_session = onnxruntime.InferenceSession("vision_encoder.onnx")
embed_session = onnxruntime.InferenceSession("embed_tokens.onnx")
decoder_session = onnxruntime.InferenceSession("decoder_model_merged.onnx")

## Set config values
num_key_value_heads = config.text_config.num_key_value_heads
head_dim = config.text_config.head_dim
num_hidden_layers = config.text_config.num_hidden_layers
eos_token_id = config.text_config.eos_token_id
image_token_id = config.image_token_id


# 2. Prepare inputs
## Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]

## Load image and apply processor
image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="np")

## Prepare decoder inputs
batch_size = inputs['input_ids'].shape[0]
past_key_values = {
    f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ('key', 'value')
}
image_features = None
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
position_ids = np.arange(inputs['input_ids'].shape[1]).reshape(1, -1)


# 3. Generation loop
max_new_tokens = 1024
generated_tokens = []
for i in range(max_new_tokens):
  inputs_embeds = embed_session.run(None, {'input_ids': input_ids})[0]

  if image_features is None:
    ## Only compute vision features if not already computed
    image_features = vision_session.run(
        ['image_features'],  # List of output names or indices
        {
            'pixel_values': inputs['pixel_values'],
            'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
        }
    )[0]
    
    ## Merge text and vision embeddings
    inputs_embeds[inputs['input_ids'] == image_token_id] = image_features.reshape(-1, image_features.shape[-1])

  print("inputs_embeds shape:", inputs_embeds.shape)
  print("attention_mask shape:", attention_mask.shape)
  print("position_ids shape:", position_ids.shape)

  logits, *present_key_values = decoder_session.run(None, dict(
      inputs_embeds=inputs_embeds,
      attention_mask=attention_mask,
      position_ids=position_ids,
      **past_key_values,
  ))

  ## Update values for next generation loop
  next_token = logits[:, -1].argmax(-1, keepdims=True)
  generated_tokens.append(next_token[0, 0])
  input_ids = next_token
  attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids)], axis=-1)
  position_ids = np.concatenate([position_ids, np.array([[position_ids[0, -1] + 1]])], axis=-1)
  for j, key in enumerate(past_key_values):
    past_key_values[key] = present_key_values[j]

  if (next_token == eos_token_id).all():
    break

  ## (Optional) Streaming
  print(processor.decode(next_token[0]), end='')
print()

# 4. Output result
print(processor.batch_decode([generated_tokens], skip_special_tokens=True))
