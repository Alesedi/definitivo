// GUIDA: MongoDB Atlas Trigger per Auto-Delete Votazioni
// 
// STEP 1: Vai su MongoDB Atlas Dashboard
// STEP 2: Database → Triggers → "Add Trigger"
// STEP 3: Configura il trigger:

/*
=== CONFIGURAZIONE TRIGGER ===
Trigger Name: auto_delete_user_votes
Trigger Type: Database
Cluster: [seleziona il tuo cluster]
Database: [nome del tuo database]
Collection: utenti
Operation Type: Delete
Full Document: false
Event Ordering: false

=== FUNCTION CODE (copia questo codice nel trigger) ===
*/

exports = function(changeEvent) {
    try {
        // Ottieni l'ID dell'utente cancellato
        const deletedUserId = changeEvent.documentKey._id;
        
        console.log(`🗑️ Trigger attivato: cancellazione utente ${deletedUserId}`);
        
        // Connetti alla collection votazioni
        const votazioni = context.services.get("mongodb-atlas")
            .db(changeEvent.ns.db)
            .collection("votazioni");
        
        // Cancella tutte le votazioni dell'utente
        return votazioni.deleteMany({"utente": deletedUserId})
            .then(result => {
                console.log(`✅ Auto-cancellate ${result.deletedCount} votazioni per utente ${deletedUserId}`);
                return {
                    success: true,
                    deletedVotes: result.deletedCount,
                    userId: deletedUserId
                };
            })
            .catch(error => {
                console.error(`❌ Errore auto-cancellazione votazioni:`, error);
                throw error;
            });
            
    } catch (error) {
        console.error(`❌ Errore nel trigger:`, error);
        return {
            success: false,
            error: error.message
        };
    }
};

/*
=== AFTER CREATING THE TRIGGER ===
1. Save & Deploy the trigger
2. Enable the trigger (make sure it's "Enabled")
3. Test: Delete a user from MongoDB Compass
4. Check: Votes should be auto-deleted!

=== TESTING ===
- Delete user from Compass → Votes auto-deleted ✅
- Delete user from API → Votes auto-deleted ✅  
- Delete user from Shell → Votes auto-deleted ✅
*/