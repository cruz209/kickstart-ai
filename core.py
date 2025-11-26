# from __future__ import annotations
#
# import os
# import json
# from typing import Dict, Optional
#
# from liftoff.utils.intent_parser import parse_intent
# from liftoff.utils.meta_prompt import build_meta_prompt
# from liftoff.utils.file_writer import write_file_tree
#
# # Backends
# from .backends.openai_backend import OpenAIBackend
# from .backends.hf_backend import HuggingFaceBackend
#
#
# class LiftOff:
#     """
#     Core orchestration engine for LiftOff.
#
#     Responsibilities:
#         - Parse user intent
#         - Build meta-prompt describing the project
#         - Select and call an LLM backend
#         - Receive file tree JSON
#         - Write scaffolded project to disk
#     """
#
#     def __init__(
#         self,
#         api_key: Optional[str] = None,
#         hf_token: Optional[str] = None,
#         output_dir: str = "liftoff_output",
#     ) -> None:
#         self.api_key = api_key or os.getenv("OPENAI_API_KEY")
#         self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
#         self.output_dir = output_dir
#
#     # ----------------------------------------------------------------------
#     # Backend selection logic
#     # ----------------------------------------------------------------------
#
#     def _choose_backend(self, metadata: Dict):
#         """
#         Backend priority:
#
#             1) OpenAIBackend      — best quality, easiest, default
#             2) HuggingFaceBackend — cloud fallback if HF token provided
#             3) Error if neither exists
#         """
#
#         # ---------- 1. OPENAI ALWAYS FIRST ----------
#         if self.api_key:
#             print("[LiftOff] Using OpenAI backend.")
#             return OpenAIBackend(api_key=self.api_key)
#
#         # ---------- 2. HuggingFace Inference API ----------
#         if self.hf_token:
#             print("[LiftOff] Using HuggingFace Inference API backend.")
#             return HuggingFaceBackend(hf_token=self.hf_token)
#
#         # ---------- 3. Nothing available ----------
#         raise RuntimeError(
#             "No available backend.\n\n"
#             "To fix, do ONE of the following:\n"
#             "  • Set OPENAI_API_KEY (recommended), OR\n"
#             "  • Set HUGGINGFACE_TOKEN to use HF Inference API.\n"
#         )
#
#     # ----------------------------------------------------------------------
#     # Main project generation flow
#     # ----------------------------------------------------------------------
#
#     def create(self, prompt: str, output_dir: Optional[str] = None) -> None:
#         """
#         High-level creation flow:
#             1. Parse user input → metadata
#             2. Build meta-prompt for the backend
#             3. Run backend.generate_project(...)
#             4. Write resulting file tree to disk
#         """
#
#         print("[LiftOff] Parsing intent...")
#         metadata = parse_intent(prompt)
#
#         print("[LiftOff] Selecting backend...")
#         backend = self._choose_backend(metadata)
#
#         print("[LiftOff] Building meta-prompt...")
#         meta_prompt = build_meta_prompt(prompt, metadata)
#
#         print("[LiftOff] Generating project from backend...")
#         file_tree = backend.generate_project(meta_prompt, metadata)
#
#         # Choose output directory
#         root = output_dir or self.output_dir
#
#         print(f"[LiftOff] Writing project files to: {root}")
#         write_file_tree(file_tree, root)
#
#         print("[LiftOff] Project created successfully.")
from __future__ import annotations

import os
from typing import Dict, Optional

# Correct paths based on your codebase structure
from liftoff.utils.intent_parser import parse_intent
from liftoff.utils.meta_prompt import build_meta_prompt
from liftoff.utils.file_writer import write_file_tree
from liftoff.utils.validator import validate_file_tree

# Backends
from .backends.openai_backend import OpenAIBackend
from .backends.hf_backend import HuggingFaceBackend


class LiftOff:
    """
    Core orchestration engine for LiftOff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        hf_token: Optional[str] = None,
        output_dir: str = "liftoff_output",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        self.output_dir = output_dir

    # ----------------------------------------------------------------------
    # Backend selection logic
    # ----------------------------------------------------------------------

    def _choose_backend(self, metadata: Dict):
        """
        Backend priority:
            1. OpenAI backend
            2. HuggingFace Inference API
            3. Error
        """

        # ---------- 1. OpenAI ----------
        if self.api_key:
            print("[LiftOff] Using OpenAI backend.")
            return OpenAIBackend(api_key=self.api_key)

        # ---------- 2. HuggingFace ----------
        if self.hf_token:
            print("[LiftOff] Using HuggingFace Inference API backend.")
            return HuggingFaceBackend(hf_token=self.hf_token)

        # ---------- 3. Nothing ----------
        raise RuntimeError(
            "No available backend.\n\n"
            "Set OPENAI_API_KEY or HUGGINGFACE_TOKEN."
        )

    # ----------------------------------------------------------------------
    # Main project generation flow
    # ----------------------------------------------------------------------

    def create(self, prompt: str, output_dir: Optional[str] = None) -> None:
        """
        Creation pipeline:
            • parse → metadata
            • meta-prompt → backend
            • generate → file_tree
            • validate → warn
            • write → disk
        """

        print("[LiftOff] Parsing intent...")
        metadata = parse_intent(prompt)

        print("[LiftOff] Selecting backend...")
        backend = self._choose_backend(metadata)

        print("[LiftOff] Building meta-prompt...")
        meta_prompt = build_meta_prompt(prompt, metadata)

        print("[LiftOff] Generating project from backend...")
        file_tree = backend.generate_project(meta_prompt, metadata)

        print("[LiftOff] Validating scaffold...")
        issues = validate_file_tree(file_tree)
        if issues:
            print("⚠️  Validator warnings:")
            for issue in issues:
                print("   -", issue)

        # Choose output directory
        root = output_dir or self.output_dir

        print(f"[LiftOff] Writing project files to: {root}")
        write_file_tree(file_tree, root)

        print("[LiftOff] Project created successfully.")
