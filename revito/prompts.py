def set_prompts(language: str = "fr") -> dict:
    """Sets chatbot texts & prompts."""
    if language == "fr":
        return _set_prompts_fr()
    else:
        raise ValueError(f"Language {language} not supported.")


def _set_prompts_fr() -> dict:
    """Sets french system prompts."""
    # partner prompts
    partner_input = "Décrivez la thématique que vous souhaitez réviser : "
    partner_context = """
Tu es un partenaire de révision.
A partir d'un cours à apprendre, tu dois formuler une question dont la réponse se trouve dans ledit cours.
Tu dois respecter les règles suivantes :
- Tes questions doivent être claires, concises et simples à comprendre.
- Tes questions doivent trouver leurs réponses dans le cours donné.
- Tu ne dois poser qu'une seule question à chaque fois.
    """
    partner_context_builder = lambda u, v: f"{u}\n\n###\n\nCours à apprendre:\n{v}"
    partner_query_builder = lambda u: f"Pose une question sur la thématique suivante: {u}"

    # teacher prompts
    teacher_input = "Tapez votre réponse : "
    teacher_context = """
Tu es un professeur.
Tu dois évaluer la qualité et la pertinence des réponses de ton étudiant.
Tu dois en premier lieu fournir une note de 0 à 5 à sa réponse, puis fournir une explication de ta note.
Tu dois respecter les règles suivantes :
- Sois clair : tes évaluations doivent être claires, concises et simples à comprendre.
- Sois strict : tes notes doivent refléter la qualité de la réponse, et donc être élevée quand la réponse est bonne et basse si elle ne l'est pas.
- Sois positif : tes explications doivent être positives et encourager ton étudiant à progresser.
    """
    teacher_context_builder = lambda u, v: f"{u}\n\n###\n\nQuestion posée : {v}"
    teacher_query_builder = lambda u: f"Réponse de l'étudiant : {u}"

    # other prompts
    command_input = "Entrez une commande ('q' pour quitter, 'c' pour changer le thème, rien pour continuer): "
    goodbye_prompt = "Au revoir !"

    return {
        "partner": {
            "input": partner_input,
            "context": partner_context,
            "context_builder": partner_context_builder,
            "query_builder": partner_query_builder,
        },
        "teacher": {
            "input": teacher_input,
            "context": teacher_context,
            "context_builder": teacher_context_builder,
            "query_builder": teacher_query_builder,
        },
        "app": {
            "command": command_input,
            "goodbye": goodbye_prompt,
        }
    }
