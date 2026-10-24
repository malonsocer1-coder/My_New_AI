"""Clasificación ligera de la variante del español (México, España, otras).

Las fuentes abiertas (FineWeb-2, Wikipedia) no separan variantes regionales.
Este clasificador cuenta marcadores léxicos y gramaticales ponderados
(vocabulario, voseo/vosotros, instituciones, topónimos y moneda) para poder
enrutar el español de México a un dominio propio (``es_mx``) y darle más peso
en la mezcla. Es una heurística transparente y auditable, no un modelo:
prioriza la precisión (ante la duda, "es-neutral").
"""

from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[a-záéíóúüñ]+", re.UNICODE)

MX = {
    # vocabulario y expresiones
    "ahorita": 3, "platicar": 2, "platicamos": 2, "chamba": 3, "chido": 3, "chida": 3, "órale": 3, "güey": 3,
    "neta": 2, "padrísimo": 3, "computadora": 1, "celular": 1, "carro": 1, "alberca": 3, "recámara": 3,
    "chamarra": 3, "popote": 3, "banqueta": 3, "cajuela": 3, "regadera": 1, "elote": 2, "jitomate": 3,
    "camión": 1, "estacionamiento": 1, "boleto": 1, "lana": 1, "cuate": 2, "fresa": 0, "chavo": 2, "chava": 2,
    "mande": 2, "híjole": 3, "pinche": 2, "chamaco": 2, "tianguis": 3, "abarrotes": 2, "colonia": 1,
    # instituciones y realidades mexicanas
    "cdmx": 3, "imss": 3, "issste": 3, "curp": 3, "rfc": 2, "ine": 1, "sat": 1, "sep": 1, "pemex": 3, "cfe": 2,
    "unam": 3, "ipn": 2, "alcaldía": 2, "delegación": 1, "zócalo": 2, "mexicano": 2, "mexicana": 2,
    "mexicanos": 2, "méxico": 2, "chilango": 3, "chilanga": 3, "peso": 0, "pesos": 1,
    # estados y ciudades
    "jalisco": 3, "guadalajara": 2, "monterrey": 3, "puebla": 2, "oaxaca": 3, "chiapas": 3, "yucatán": 3,
    "veracruz": 3, "sonora": 2, "chihuahua": 3, "tamaulipas": 3, "michoacán": 3, "querétaro": 3,
    "guanajuato": 3, "sinaloa": 3, "coahuila": 3, "morelos": 2, "tabasco": 3, "zacatecas": 3, "durango": 2,
    "nayarit": 3, "aguascalientes": 3, "tlaxcala": 3, "campeche": 3, "tijuana": 3, "mérida": 1, "cancún": 3,
    "acapulco": 3, "toluca": 3, "culiacán": 3, "hermosillo": 3,
}
ES = {
    "vosotros": 3, "vosotras": 3, "vuestro": 3, "vuestra": 3, "vuestros": 3, "habéis": 3, "tenéis": 3,
    "sois": 3, "queréis": 3, "podéis": 3, "estáis": 3, "hacéis": 3, "ordenador": 3, "móvil": 2, "coche": 2,
    "zumo": 3, "patata": 3, "patatas": 3, "aparcamiento": 3, "billete": 1, "gilipollas": 3, "guay": 3,
    "mola": 3, "vale": 1, "coger": 1, "euros": 2, "euro": 1, "españa": 2, "madrid": 2, "barcelona": 2,
    "ayuntamiento": 2, "autonómica": 3, "autonómico": 3, "comunidad": 0, "piso": 1, "tío": 1, "chaval": 3,
    "valencia": 1, "sevilla": 2, "andalucía": 3, "cataluña": 3, "galicia": 3,
}
OTHER = {
    "vos": 2, "tenés": 3, "querés": 3, "sos": 2, "podés": 3, "che": 2, "pibe": 3, "boludo": 3, "argentina": 2,
    "argentino": 2, "buenos": 0, "colombia": 2, "colombiano": 2, "chile": 2, "chileno": 2, "perú": 2,
    "peruano": 2, "venezuela": 2, "uruguay": 2, "bogotá": 2, "lima": 1, "santiago": 1, "parcero": 3, "bacán": 3,
}


def dialect_scores(text: str, max_words: int = 3000) -> dict[str, float]:
    words = _WORD.findall(text.lower()[: max_words * 8])[:max_words]
    if not words:
        return {"es-MX": 0.0, "es-ES": 0.0, "es-otros": 0.0}
    c = Counter(words)
    per_k = 1000 / len(words)
    return {"es-MX": sum(c[w] * s for w, s in MX.items()) * per_k,
            "es-ES": sum(c[w] * s for w, s in ES.items()) * per_k,
            "es-otros": sum(c[w] * s for w, s in OTHER.items()) * per_k}


def classify_dialect(text: str, min_score: float = 4.0, margin: float = 1.5) -> str:
    """Devuelve 'es-MX', 'es-ES', 'es-otros' o 'es-neutral'.

    Exige una puntuación mínima por cada mil palabras y que la variante ganadora
    supere a las demás por un factor ``margin`` (ante la duda, neutral).
    """
    s = dialect_scores(text)
    best, val = max(s.items(), key=lambda kv: kv[1])
    others = max(v for k, v in s.items() if k != best)
    if val >= min_score and val >= margin * others:
        return best
    return "es-neutral"
