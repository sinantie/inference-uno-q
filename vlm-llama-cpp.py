"""
Example for using SmolVLM-256M-Instruct-GGUF with llama.cpp OpenAI-compatible server.

To run this example:
1. Make sure you have llama.cpp built with server support (llama-server should be in build/bin/)
2. Install dependencies: pip install requests
3. Start the server from the llama.cpp directory:
   cd /path/to/llama.cpp && ./build/bin/llama-server -hf ggml-org/SmolVLM-256M-Instruct-GGUF --port 8080
4. Run this script with an image file:
   python vlm-llama-cpp.py /path/to/image.jpg -p "Describe this image" -m 256
"""

import requests
import base64
from pathlib import Path
import argparse
import sys
from PIL import Image, ImageOps
from io import BytesIO


def image_to_data_uri(path, size=(256, 256), quality=80):
    # Load and auto-orient
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    # Resize (preserve aspect ratio, then crop to exact size)
    img.thumbnail(size, Image.LANCZOS)
    if img.size != size:
        w, h = img.size
        tw, th = size
        left = max(0, (w - tw) // 2)
        top = max(0, (h - th) // 2)
        img = img.crop((left, top, left + tw, top + th))

    # Compress to JPEG in memory
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')


def encode_image_to_base64(image_path, resize=False):
    """Encode an image file to base64 string."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as image_file:
        img = Image.open(image_file).convert("RGB")
        if resize:
            print("Resizing image to 256x256")
            img = img.resize((256, 256))  # match model spec
        
        # Save image to bytes
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

def chat_with_image(image_path, prompt, max_tokens=32, resize=False, server_url="http://localhost:8080"):
    """Send a chat completion request with an image to the llama.cpp server."""

    # Encode the image
    # base64_image = encode_image_to_base64(image_path, resize=resize)
    base64_image = image_to_data_uri(image_path, size=(256, 256), quality=80)

    # Prepare the request payload
    payload = {
        "model": "ggml-org/SmolVLM-256M-Instruct-GGUF",  # or whatever alias you set
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    # Make the request
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with SmolVLM using images")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("-r", "--resize", action="store_true",
                       help="Resize image to 256x256 before processing (default: False)")
    parser.add_argument("-p", "--prompt", default="Describe this image in detail",
                       help="Text prompt to send with the image (default: 'Describe this image in detail')")
    parser.add_argument("-m", "--max-tokens", type=int, default=512,
                       help="Maximum number of tokens to generate (default: 512)")
    parser.add_argument("-u", "--url", default="http://localhost:8080",
                       help="Server URL (default: http://localhost:8080)")

    args = parser.parse_args()

    try:
        response = chat_with_image(args.image_path, args.prompt, args.max_tokens, args.url)
        print("Assistant:", response)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the llama.cpp server is running with SmolVLM model:")
        print("cd /path/to/llama.cpp && ./build/bin/llama-server -hf ggml-org/SmolVLM-256M-Instruct-GGUF --port 8080")
        print()
        print("Usage example:")
        print("python vlm-llama-cpp.py /path/to/image.jpg -p 'What do you see?' -m 256")
        sys.exit(1)
