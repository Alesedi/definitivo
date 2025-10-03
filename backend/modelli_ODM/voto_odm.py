from mongoengine import Document, IntField, ReferenceField

class Votazione(Document):
    utente = ReferenceField("Utente", required=True)  # usa stringa per evitare circular import
    film = ReferenceField("Film", required=True)      # usa stringa per evitare circular import
    valutazione = IntField(min_value=1, max_value=5, required=True)

    meta = {
        "collection": "votazioni",
        "indexes": [
            {"fields": ("utente", "film"), "unique": True}  # evita duplicati utente-film
        ]
    }
