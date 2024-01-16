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
    # retriever prompts
    retriever_input = "Décrivez la thématique que vous souhaitez réviser : "

    # partner prompts
    partner_context = """
Tu es un partenaire de révision.
A partir d'un corpus de documents, tu dois formuler une question dont la réponse se trouve dans ledit corpus.
Tu dois respecter les règles suivantes :
- Tes questions doivent être claires, concises et simples à comprendre.
- Tes questions doivent trouver leurs réponses dans le cours donné.
- Tu ne dois poser qu'une seule question à chaque fois.
    """
    partner_query_builder = lambda: "Pose moi une question sur le cours"

    # teacher prompts
    teacher_context = """
Tu es un professeur.
Tu dois évaluer la qualité et la pertinence des réponses de ton étudiant.
Tu dois en premier lieu fournir une note de 0 à 5 à sa réponse, puis fournir une explication de ta note.
Tu dois respecter les règles suivantes :
- Sois clair : tes évaluations doivent être claires, concises et simples à comprendre.
- Sois strict : tes notes doivent refléter la qualité de la réponse, et donc être élevée quand la réponse est bonne et basse si elle ne l'est pas.
- Sois positif : tes explications doivent être positives et encourager ton étudiant à progresser.
    """
    teacher_input = "Tapez votre réponse :"
    teacher_query_builder = lambda u, v: f"Question : {u}\n Réponse : {v}"

    # other prompts
    command_input = "Entrez une commande ('q' pour quitter, 'c' pour changer le thème, rien pour continuer): "
    goodbye_prompt = "Au revoir !"

    return {
        "retriever": {
            "input": retriever_input,
        },
        "partner": {
            "context":  partner_context,
            "query_builder": partner_query_builder,
        },
        "teacher": {
            "input":  teacher_input,
            "context": teacher_context,
            "query_builder": teacher_query_builder,
        },
        "app": {
            "command": command_input,
            "goodbye": goodbye_prompt,
        }
    }


def _set_prompts_en() -> dict:
    """Sets english system prompts."""
    # retriever prompts
    retriever_input = "Describe the section you want to revise: "

    # partner prompts
    partner_context = """
You are a partner of revision.
From the provided course, you must formulate a question to your partner allowing you to check its comprehension of the course.
You must respect the following rules :
- Your questions must be clear, concise and simple to understand.
- Your questions must find their answers in the course given.
- You must only ask one question at a time.
    """
    partner_query_builder = lambda: "Ask me a question about the course"

    # teacher prompts
    teacher_context = """
You are a teacher.
You must evaluate the quality and pertinence of the student's responses.
You must first provide a grade of 0 to 5 to her response, then provide an explanation of her grade.
You must respect the following rules :
- Be clear : your evaluations must be clear, concise and simple to understand.
- Be strict : your grades must reflect the quality of the response, and must be high when the response is good and low if it is not.
- Be positive : your explanations must be positive and encourage your student to progress.
    """
    teacher_input = "Enter your response: "
    teacher_query_builder = lambda u, v: f"Question : {u}\n Response : {v}"

    # other prompts
    command_input = "Enter a command ('q' to quit, 'c' to change theme, nothing to continue): "
    goodbye_prompt = "Bye !"

    return {
        "retriever": {
            "input": retriever_input,
        },
        "partner": {
            "context":  partner_context,
            "query_builder": partner_query_builder,
        },
        "teacher": {
            "input":  teacher_input,
            "context": teacher_context,
            "query_builder": teacher_query_builder,
        },
        "app": {
            "command": command_input,
            "goodbye": goodbye_prompt,
        }
    }
