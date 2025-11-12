import os
import json
import time
import logging
from typing import Optional, Dict, Any, List
import requests

logger = logging.getLogger(__name__)


class OMDbService:
    """Client minimale per OMDb API con caching file-based.

    Impostare la chiave OMDB_API_KEY come variabile d'ambiente oppure passare la chiave
    al costruttore. Per semplicità in sviluppo si può impostare direttamente la chiave
    (non raccomandato per produzione).
    """

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        # default key fornita dall'utente se non presente in env
        provided_key = api_key or os.environ.get('OMDB_API_KEY') or '2639fb0f'
        self.api_key = provided_key
        self.base_url = 'http://www.omdbapi.com/'
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '..', 'cache', 'omdb')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.rate_limit_sleep = 0.25

    def _cache_path(self, key: str) -> str:
        # Sanitize key to a safe filename (remove characters invalid on Windows and limit length)
        import re
        # replace slashes and spaces first
        safe = key.replace('/', '_').replace(' ', '_')
        # remove invalid filename characters <>:"/\\|?*
        safe = re.sub(r'[<>:"\\/\|\?\*]', '_', safe)
        # collapse multiple underscores
        safe = re.sub(r'_+', '_', safe)
        # trim leading/trailing underscores/dots
        safe = safe.strip('_.')
        # limit filename length to avoid filesystem issues
        max_len = 200
        if len(safe) > max_len:
            safe = safe[:max_len]
        return os.path.join(self.cache_dir, f"{safe}.json")

    def _load_cache(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, data: Dict[str, Any]):
        path = self._cache_path(key)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"OMDb: impossibile salvare cache {path}: {e}")

    def _call_api(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            logger.error('OMDb API key non configurata')
            return None
        params = dict(params)
        params['apikey'] = self.api_key
        try:
            resp = requests.get(self.base_url, params=params, timeout=8)
            time.sleep(self.rate_limit_sleep)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('Response') == 'True' or 'Search' in data:
                    return data
                else:
                    logger.debug(f"OMDb risposta negativa: {data.get('Error')}")
                    return None
            else:
                logger.warning(f"OMDb HTTP {resp.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Errore chiamata OMDb: {e}")
            return None

    def get_movie_by_imdb(self, imdb_id: str) -> Optional[Dict[str, Any]]:
        key = f"imdb_{imdb_id}"
        cached = self._load_cache(key)
        if cached:
            return cached
        data = self._call_api({'i': imdb_id, 'plot': 'full'})
        if data:
            self._save_cache(key, data)
        return data

    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
        key = f"search_{title}_{year or 'any'}"
        cached = self._load_cache(key)
        if cached:
            return cached
        params = {'s': title}
        if year:
            params['y'] = str(year)
        data = self._call_api(params)
        if data and 'Search' in data and len(data['Search']) > 0:
            first = data['Search'][0]
            self._save_cache(key, first)
            return first
        return None

    def get_poster_url(self, omdb_data: Dict[str, Any]) -> Optional[str]:
        if not omdb_data:
            return None
        poster = omdb_data.get('Poster')
        if poster and poster != 'N/A':
            return poster
        return None

    def bulk_fetch_by_imdb(self, imdb_ids: List[str]) -> List[Dict[str, Any]]:
        results = []
        for i, imdb_id in enumerate(imdb_ids):
            data = self.get_movie_by_imdb(imdb_id)
            if data:
                results.append(data)
        return results


omdb_service = OMDbService()
