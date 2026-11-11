"""
The 5 models under test, as OpenRouter slugs.
Kept as a list of (display_name, slug) so output tables show friendly names.
"""
MODELS = [
    ("GPT-5.6 Terra", "openai", "gpt-5.6-terra"),
    ("Gemini Flash Lite", "gemini", "gemini-3.5-flash-lite"),
]


if __name__ == "__main__":
    print(f"{len(MODELS)} models under test:")
    for name, slug in MODELS:
        print(f"  {name:18s} -> {slug}")