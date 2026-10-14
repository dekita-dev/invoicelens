"""InvoiceLens — local-first invoice and receipt triage for sole proprietors.

Everything runs on this machine. No API key, no network call, no client data
leaving the laptop. The field extraction is done by an open-weight model
(Gemma 2 2B, GGUF) served through the llama.cpp runtime bundled with gpt4all.
"""

__version__ = "0.1.0"
