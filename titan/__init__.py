"""TITAN AI: modelo de lenguaje propio entrenado desde pesos aleatorios.

Paquetes:
    titan.model      Arquitectura del transformer (densa y Mixture of Experts).
    titan.tokenizer  Entrenamiento y uso del tokenizador BPE propio.
    titan.data       Limpieza, deduplicación, descontaminación y shards.
    titan.train      Bucle de entrenamiento, checkpoints y paralelismo.
    titan.sft        Ajuste supervisado de instrucciones y uso de herramientas.
    titan.eval       Benchmarks reproducibles.
    titan.infer      Generación con KV cache y servidor HTTP.
    titan.export     Exportación, verificación y cuantización.
    titan.agent      Protocolo de herramientas, permisos, sandbox y memoria.
    titan.cloud      Estimación de costes, proveedores GPU y guardián de gasto.
"""

__version__ = "0.1.0"
