from typing import Dict, Any
import json
import os

import transformers
import torch

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None


# --- DEFAULT MODEL: Local Mistral ---
LOCAL_MODEL = "microsoft/Phi-3-mini-4k-instruct"



class HuggingFaceBackend:
    """
    Local-first backend for Autoforge/Liftoff.

    Logic:
      1. Try local transformers inference (no token needed).
      2. If that fails AND user has HF_TOKEN, fallback to HF hosted inference.
      3. Otherwise raise clean error telling user to install transformers or model.
    """

    def __init__(self, hf_token: str | None = None, model_id: str | None = None) -> None:
        self.model_id = model_id or LOCAL_MODEL
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")

        self.local_pipeline = None
        self.client = None

        # --- Try LOCAL first ---
        try:
            print("[Autoforge] Loading local model:", self.model_id)
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.local_pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                device_map="auto",
                torch_dtype=torch_dtype,
            )

            print("[Autoforge] Local model loaded successfully.")

        except Exception as e:
            print("[Autoforge] Local model failed to load:", e)

            # --- Try HF remote only if token exists ---
            if self.hf_token and InferenceClient is not None:
                print("[Autoforge] Falling back to HuggingFace Inference API.")
                self.client = InferenceClient(self.model_id, token=self.hf_token)
            else:
                raise RuntimeError(
                    f"Could not load local model '{self.model_id}'.\n"
                    f"Install it via:  transformers, accelerate, bitsandbytes\n"
                    f"Or set HUGGINGFACE_TOKEN for API fallback."
                )

    # ----------------------------------------------------------------------

    def _run_local(self, prompt: str) -> str:
        """Run inference with local transformers pipeline."""
        out = self.local_pipeline(
            prompt,
            max_new_tokens=1024,
            do_sample=False,
        )
        # Transformers returns: [{'generated_text': "..."}]
        return out[0]["generated_text"]

    def _run_hf(self, prompt: str) -> str:
        """Run inference with HuggingFace API (fallback)."""
        try:
            resp = self.client.text_generation(
                prompt,
                max_new_tokens=1024,
                temperature=0.2,
            )
            if isinstance(resp, str):
                return resp
            return resp.get("generated_text", str(resp))
        except Exception:
            resp = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return resp.choices[0].message["content"]

    # ----------------------------------------------------------------------

    def generate_project(self, meta_prompt: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        This is the main entry point used by Autoforge/Liftoff.

        Returns parsed JSON, or raises a readable error.
        """
        # --- 1. Prefer local ---
        if self.local_pipeline is not None:
            raw = self._run_local(meta_prompt)
        else:
            raw = self._run_hf(meta_prompt)

        # --- 2. Parse JSON ---
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model did not return valid JSON.\n"
                f"Error: {e}\n\nRaw output:\n{raw[:400]}..."
            )
