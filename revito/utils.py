from typing import Optional, Dict


def build_system_prompt(
        system: Optional[str] = "",
        sections: Optional[Dict[str, str]] = None
) -> str:
    """
    Builds the system prompt for a given LLM.
    """
    if sections is None:
        sections = {}
    prompt = system
    for key, section in sections.items():
        prompt += f"\n\n### {key.upper().strip()}\n{section}"
    return prompt
