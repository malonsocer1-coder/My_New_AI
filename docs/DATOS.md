# Datos de entrenamiento

## Principios

1. **Solo fuentes con licencia declarada.** `SourceSpec.license` es obligatorio;
   el pipeline rechaza fuentes sin licencia salvo autorización explícita
   (`allow_unknown_license: true`).
2. **Trazabilidad.** `data_card.json` registra fuentes, licencias, origen,
   motivos de rechazo por fuente y tokens por dominio; `manifest.json` fija
   el SHA-256 de cada shard y del tokenizador.
3. **Separación estricta de evaluación.** Todo benchmark se indexa (13-gramas
   y coincidencias cortas con límites de palabra) antes de limpiar; cualquier
   documento que coincida se descarta. `titan data audit` lo verifica a posteriori.
4. **Sintéticos verificados por programa** y con semillas disjuntas de las de evaluación.

## Fases del pipeline (`titan data prepare`)

| Fase | Implementación | Notas |
|---|---|---|
| Carga | `zip_url` (con SHA-256), `local_glob`, `jsonl`, `hf` (streaming), `synthetic` | `hf` solo donde haya acceso a huggingface.co |
| Normalización | NFC, saltos de línea, caracteres de control, espacios | |
| Privacidad | correos, teléfonos, IPs y secretos → marcadores | `scrub_pii: true` por fuente |
| Calidad (prosa) | longitud, longitud media de palabra, fracción alfabética, símbolos, líneas duplicadas, viñetas, repeticiones | heurísticas tipo Gopher/C4 |
| Calidad (código) | marcadores de autogenerado, líneas largas, fracción alfanumérica | |
| Idioma | identificador por palabras vacías (es/en/pt/fr), sin modelos externos | |
| Deduplicación | exacta (SHA-256 normalizado) + MinHash-LSH (5-gramas, 128 permutaciones) | global, antes de partir train/val |
| Descontaminación | 13-gramas + textos cortos de los benchmarks | |
| Partición | por hash del documento (determinista) | |
| Tokenizador | BPE propio, muestra equilibrada por dominio | se reutiliza si ya existe |
| Shards | binarios uint16 (vocab ≤ 65 536) o uint32 + SHA-256 | lectura por memmap |

## Corpus de la Etapa A (real, en esta sesión)

| Dominio | Fuente | Licencia | Docs train | Tokens train |
|---|---|---|---|---|
| es | Europarl (muestra NLTK) + DUDH | sin restricciones conocidas / dominio público | 1 063 | 871 430 |
| en | Europarl + Project Gutenberg (NLTK) | sin restricciones conocidas / dominio público | 3 770 | 4 236 634 |
| multi | Europarl pt/fr/it | sin restricciones conocidas | 3 264 | 2 761 998 |
| code | biblioteca estándar de CPython | PSF-2.0 | 447 | 2 530 794 |
| math | aritmética sintética verificada | propia | 2 547 | 838 045 |

Rechazos registrados: 1 duplicado exacto (DUDH en español aparece dos veces en
la fuente), 54 módulos de Python casi duplicados, 7 autogenerados, 15 por
longitud y 378 documentos sintéticos de aritmética que compartían pares de
números con el conjunto de evaluación (criterio conservador: se descarta aunque
cambie el operador). La auditoría posterior encontró **0** documentos contaminados.

## Mezclas para el PC y la nube

* `configs/data/pc_es.yaml`: ~2,5–3 mil millones de tokens, mayoritariamente
  español (FineWeb-2 `spa_Latn`, Wikipedia `es`), más FineWeb-Edu, OpenWebMath,
  aritmética sintética y código.
* `configs/data/cloud_main.yaml`: cientos de miles de millones de tokens para 1–8 B.

Fuentes recomendadas y sus licencias:

| Fuente | Contenido | Licencia |
|---|---|---|
| HuggingFaceFW/fineweb-2 (`spa_Latn`, `por_Latn`, …) | web multilingüe filtrada y deduplicada | ODC-By 1.0 + términos de Common Crawl |
| HuggingFaceFW/fineweb-edu | web educativa en inglés | ODC-By 1.0 |
| wikimedia/wikipedia | enciclopedia | CC BY-SA 4.0 / GFDL (atribución) |
| open-web-math/open-web-math | matemáticas de la web | ODC-By 1.0 |
| bigcode/the-stack-dedup | código con licencias por archivo | términos de BigCode: requiere aceptación |
| CPython stdlib | código Python | PSF-2.0 |

**Español de México.** FineWeb-2 y Wikipedia no separan variantes regionales.
`titan/data/dialect.py` clasifica cada documento en español como es-MX, es-ES,
otras variantes o neutral, con marcadores léxicos y gramaticales ponderados:
- vocabulario («ahorita», «alberca», «computadora» frente a «ordenador», «zumo»);
- formas verbales (vosotros, voseo);
- instituciones (IMSS, CURP, SEP), topónimos y moneda.

Es una heurística transparente que ante la duda responde «neutral». Con
`dialect_routing: true` en la fuente, los textos mexicanos van al dominio
`es_mx`, que recibe su propio peso (`extra_domain_weights`, p. ej. 0,20 en el
PC); `data_card.json` registra cuántos documentos hay de cada variante. Para
reforzarlo más se pueden añadir fuentes mexicanas con licencia verificada:
datos abiertos del Gobierno de México (datos.gob.mx, Licencia de Libre Uso MX),
publicaciones oficiales de dominio público y textos con permiso expreso.

**Escala:** el pipeline es de un solo proceso por diseño (reproducible y fácil
de auditar). Para más de ~100 000 M tokens conviene ejecutarlo en una máquina
CPU barata (o en el PC) y subir los shards al almacenamiento remoto antes de
alquilar GPUs; las GPU no deben esperar a los datos.

## Datos de ajuste (SFT)

`scripts/make_sft_data.py` genera conversaciones con **ejecución real** de las
herramientas (calculadora, archivos, Python en sandbox, búsqueda local,
recuperación de errores, permisos denegados, saludos y preguntas sin
respuesta). `titan sft prepare` las empaqueta sin partir conversaciones y con
pérdida solo en los turnos del asistente. Para conversación general se pueden
añadir conjuntos abiertos con licencia permisiva (p. ej. OASST2 o Aya,
Apache-2.0), convertidos al formato `{"messages": [...]}`.
