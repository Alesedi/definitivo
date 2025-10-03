# AFlix - Sistema di Raccomandazioni Cinematografiche

Un'applicazione web completa per raccomandazioni di film personalizzate, costruita con FastAPI (backend) e React (frontend).

## 🎬 Caratteristiche

- **Registrazione e Login** sicuri con JWT
- **Onboarding intelligente** con selezione generi preferiti
- **Sistema di votazione** per film già visti
- **Raccomandazioni personalizzate** basate su ML
- **Design moderno** con palette blu e arancione
- **Card film dettagliate** con poster, trama e informazioni

## 🏗️ Struttura del Progetto

```
fastApiProject/
├── backend/                 # API FastAPI
│   ├── main.py             # Entry point dell'API
│   ├── database/           # Configurazione MongoDB
│   ├── modelli_ODM/        # Modelli MongoDB
│   ├── modello/            # Modelli Pydantic
│   ├── route/              # Endpoint API
│   └── service/            # Logica business
│
└── frontend/               # App React
    ├── public/             # File statici
    ├── src/
    │   ├── components/     # Componenti React
    │   ├── pages/          # Pagine principali
    │   ├── services/       # API calls
    │   ├── styles/         # Styling e tema
    │   └── App.js          # Componente principale
    └── package.json
```

## 🚀 Avvio del Progetto

### Backend (FastAPI)

1. Vai nella cartella backend:
```bash
cd backend
```

2. Avvia il server:
```bash
python main.py
```

Il backend sarà disponibile su: `http://localhost:8005`

### Frontend (React)

1. Vai nella cartella frontend:
```bash
cd frontend
```

2. Installa le dipendenze:
```bash
npm install
```

3. Avvia l'app React:
```bash
npm start
```

Il frontend sarà disponibile su: `http://localhost:3000`

## 🎨 Design

### Palette Colori
- **Primario**: Blu (#1E3A8A, #3B82F6)
- **Secondario**: Arancione (#EA580C, #FB923C)
- **Neutri**: Grigi per testo e background

### Componenti UI
- Card film con hover effects
- Bottoni con animazioni
- Form design moderno
- Layout responsivo

## 📱 Funzionalità

### 1. Autenticazione
- Registrazione con validazione
- Login con gestione sessioni
- JWT tokens per sicurezza

### 2. Onboarding (Primo Login)
- Selezione generi preferiti (min 3)
- Votazione film iniziale per training

### 3. Dashboard
- Raccomandazioni personalizzate
- Statistiche utente
- Sistema di rating integrato

### 4. Sistema ML (Futuro)
- Algoritmi di raccomandazione
- Apprendimento da preferenze utente
- Filtri collaborativi

## 🔗 API Endpoints

### Autenticazione
- `POST /auth/register` - Registrazione
- `POST /auth/login` - Login

### Onboarding  
- `GET /onboarding/genres` - Lista generi TMDB
- `POST /onboarding/select-genres/{user_id}` - Salva generi
- `GET /onboarding/movies-for-rating/{user_id}` - Film per rating

### Votazioni
- `POST /ratings/vote/{user_id}` - Vota singolo film
- `POST /ratings/vote-multiple/{user_id}` - Voti multipli
- `GET /ratings/user-ratings/{user_id}` - Voti utente

### Raccomandazioni
- `GET /recommendations/for-user/{user_id}` - Raccomandazioni
- `GET /recommendations/stats/{user_id}` - Statistiche

## 🔧 Configurazione

### Variabili d'ambiente (.env)
```
MONGODB_URI=your_mongodb_connection_string
TMDB_API_KEY=your_tmdb_api_key
TMDB_BASE_URL=https://api.themoviedb.org/3
JWT_SECRET_KEY=your_secret_key
```

## 🎯 Prossimi Sviluppi

- [ ] Algoritmi ML avanzati
- [ ] Sistema di recensioni
- [ ] Condivisione social
- [ ] App mobile
- [ ] Cache e ottimizzazioni

## 🛠️ Tecnologie Utilizzate

**Backend:**
- FastAPI
- MongoDB + MongoEngine
- TMDB API
- JWT Authentication
- Pydantic

**Frontend:**
- React 18
- Styled Components
- React Router
- Axios
- React Icons

Buon sviluppo! 🚀