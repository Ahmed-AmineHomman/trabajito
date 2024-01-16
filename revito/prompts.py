def set_prompts(language: str = "fr") -> dict:
    """Sets chatbot system prompts."""
    if language == "fr":
        return _set_prompts_fr()
    elif language == "en":
        return _set_prompts_en()
    else:
        raise ValueError(f"Language {language} not supported.")


def _set_prompts_fr() -> dict:
    """Sets french system prompts."""
    partner_context = """
Tu es un partenaire de révision.
A partir d'une section de cours donnée, tu dois fournir une question à ton partenaire permettant de vérifier sa compréhension du cours.
Tu dois respecter les règles suivantes :
- Tes questions doivent être claires, concises et simples à comprendre.
- Tes questions doivent trouver leurs réponses dans le cours donné.
- Tu ne dois poser qu'une seule question à chaque fois.
    """

    teacher_context = """
Tu es un professeur.
Tu dois évaluer la qualité et la pertinence des réponses de ton étudiant.
Tu dois en premier lieu fournir une note de 0 à 5 à sa réponse, puis fournir une explication de ta note.
Tu dois respecter les règles suivantes :
- Sois clair : tes évaluations doivent être claires, concises et simples à comprendre.
- Sois strict : tes notes doivent refléter la qualité de la réponse, et doivent être élevée quand la réponse est bonne et basse si elle ne l'est pas.
- Sois positif : tes explications doivent être positives et encourager ton étudiant à progresser.
    """

    revision_query = "Pose moi une question sur le cours"
    revision_builder = lambda u: f"Question : {u}\n Réponse : "
    section_input = "Décrivez la section que vous souhaitez réviser : "
    command_input = "Entrez une commande ('q' pour quitter, 'c' pour changer le thème, rien pour continuer): "
    goodbye_prompt = "Au revoir !"

    return {
        "partner_context": partner_context,
        "teacher_context": teacher_context,
        "revision_query": revision_query,
        "revision_builder": revision_builder,
        "section_input": section_input,
        "command_input": command_input,
        "goodbye_prompt": goodbye_prompt,
    }


def _set_prompts_en() -> dict:
    """Sets english system prompts."""
    partner_context = """
You are a partner of revision.
From a given section of course given, you must provide a question to your partner allowing you to check its comprehension of the course.
You must respect the following rules :
- Your questions must be clear, concise and simple to understand.
- Your questions must find their answers in the course given.
- You must only ask one question at a time.
    """

    teacher_context = """
You are a teacher.
You must evaluate the quality and pertinence of the student's responses.
You must first provide a grade of 0 to 5 to her response, then provide an explanation of her grade.
You must respect the following rules :
- Be clear : your evaluations must be clear, concise and simple to understand.
- Be strict : your grades must reflect the quality of the response, and must be high when the response is good and low if it is not.
- Be positive : your explanations must be positive and encourage your student to progress.
    """

    revision_query = "Ask a question about the lecture"
    revision_builder = lambda u: f"Question: {u}\n Answer: "
    section_input = "Describe the section you want to revise: "
    command_input = "Enter a command ('q' to quit, 'c' to change theme, nothing to continue): "
    goodbye_prompt = "Bye !"

    return {
        "partner_context": partner_context,
        "teacher_context": teacher_context,
        "revision_query": revision_query,
        "revision_builder": revision_builder,
        "section_input": section_input,
        "command_input": command_input,
        "goodbye_prompt": goodbye_prompt,
    }
