from mongoengine import signals
from modelli_ODM.utente_odm import Utente
from modelli_ODM.voto_odm import Votazione

def cleanup_user_votes(sender, document, **kwargs):
    """Signal handler che cancella le votazioni quando un utente viene cancellato"""
    # Cancella tutte le votazioni di questo utente
    Votazione.objects(utente=document).delete()
    print(f"Cancellate tutte le votazioni per l'utente: {document.username}")

# Connetti il signal al modello Utente
signals.post_delete.connect(cleanup_user_votes, sender=Utente)