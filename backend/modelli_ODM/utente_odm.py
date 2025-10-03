from mongoengine import Document, StringField, ListField, ReferenceField, signals
from modello.enum_genere import EnumGenere

class Utente(Document):
    username = StringField(required=True, unique=True, max_length=50)
    email = StringField(required=True, unique=True)  # Rimossa validazione email rigorosa
    password = StringField(required=True)  # hashata

    # Lista di generi preferiti scelti dall'utente
    generi_preferiti = ListField(StringField(choices=[g.value for g in EnumGenere]), default=[])

    # Lista dei voti dati dall'utente
    votazioni = ListField(ReferenceField("Votazione"))

    meta = {"collection": "utenti"}

# Signal per cancellazione automatica delle votazioni
def auto_delete_user_votes(sender, document, **kwargs):
    """Cancella automaticamente tutte le votazioni di un utente quando viene eliminato"""
    from modelli_ODM.voto_odm import Votazione
    deleted_votes = Votazione.objects(utente=document).delete()
    print(f"🗑️ AUTO-CLEANUP: Cancellate {deleted_votes} votazioni per l'utente {document.username}")

# Connetti il signal
signals.post_delete.connect(auto_delete_user_votes, sender=Utente)
