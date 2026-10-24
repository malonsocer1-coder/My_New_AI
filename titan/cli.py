"""Línea de comandos de TITAN AI.

Ejemplos::

    titan data prepare --config configs/data/stage_a.yaml
    titan train --config configs/train/stage_a.yaml --set max_steps=100
    torchrun --nproc_per_node=8 -m titan.cli train --config configs/train/titan_1b.yaml
    titan generate --checkpoint runs/stage_a --prompt "Hola"
    titan eval --checkpoint runs/stage_a --suite all
    titan estimate --config configs/model/titan_8b.yaml --tokens 1e12
    titan export --checkpoint runs/stage_a --out models/base/stage_a
    titan serve --model models/base/stage_a
"""

from __future__ import annotations

import argparse
import json
import sys


def _parse_overrides(items: list[str] | None) -> dict:
    import yaml
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set espera clave=valor, recibido: {it}")
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def cmd_data(args) -> None:
    if args.action == "prepare":
        from .data.pipeline import prepare
        prepare(args.config)
    elif args.action == "audit":
        from .data.audit import audit_contamination
        print(json.dumps(audit_contamination(args.data_dir, args.eval_files), indent=2, ensure_ascii=False))


def cmd_sft(args) -> None:
    from .sft.data import prepare_sft
    print(json.dumps(prepare_sft(args.config), indent=2, ensure_ascii=False))


def cmd_train(args) -> None:
    from .train.trainer import main_from_yaml
    summary = main_from_yaml(args.config, _parse_overrides(args.set))
    if summary.get("checkpoint"):
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_generate(args) -> None:
    from .infer.engine import load_engine
    eng = load_engine(args.checkpoint, device=args.device)
    if args.chat:
        out = eng.chat([{"role": "user", "content": args.prompt}], max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature)
    else:
        out = eng.complete(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(out)


def cmd_eval(args) -> None:
    from .eval.harness import run_suite
    res = run_suite(args.checkpoint, args.suite, out_path=args.out, device=args.device, limit=args.limit,
                    data=args.data)
    print(json.dumps(res["summary"], indent=2, ensure_ascii=False))


def cmd_estimate(args) -> None:
    from .cloud.estimate import estimate_from_args
    estimate_from_args(args)


def cmd_export(args) -> None:
    from .export.exporter import export_checkpoint
    info = export_checkpoint(args.checkpoint, args.out, formats=args.formats.split(","))
    print(json.dumps(info, indent=2, ensure_ascii=False))


def cmd_quantize(args) -> None:
    from .export.quantize import quantize_model_dir
    print(json.dumps(quantize_model_dir(args.model, args.out, args.bits, args.group_size), indent=2))


def cmd_gguf(args) -> None:
    from .export import gguf_tools
    if args.action == "quantize":
        res = gguf_tools.quantize_gguf(args.model, args.out, args.type)
    else:
        res = gguf_tools.bench_gguf(args.model, args.prompt_tokens, args.gen_tokens, args.gpu_layers)
    print(json.dumps(res, indent=2))


def cmd_serve(args) -> None:
    from .ui.server import serve
    serve(args.model, host=args.host, port=args.port, device=args.device, agent_config=args.agent_config)


def cmd_grow(args) -> None:
    from .train import grow
    cfg, sd = grow.load_model_dir(args.source)
    if args.op == "depth":
        new_cfg, new_sd = grow.grow_depth(cfg, sd, args.layers, args.method)
    elif args.op == "upcycle":
        new_cfg, new_sd = grow.upcycle_to_moe(cfg, sd, args.experts, args.top_k)
    else:
        raise SystemExit(f"Operación desconocida: {args.op}")
    grow.save_model_dir(new_cfg, new_sd, args.out, {"op": args.op, "source": args.source}, args.tokenizer)
    print(json.dumps(grow.describe(new_cfg), indent=2))


def cmd_merge(args) -> None:
    from .train import grow
    loaded = [grow.load_model_dir(b) for b in args.branches]
    new_cfg, new_sd = grow.merge_branches([c for c, _ in loaded], [s for _, s in loaded], top_k=args.top_k)
    grow.save_model_dir(new_cfg, new_sd, args.out, {"op": "merge_experts", "branches": args.branches},
                        args.tokenizer)
    print(json.dumps(grow.describe(new_cfg), indent=2))


def cmd_plan(args) -> None:
    from .train.plan import Plan
    plan = Plan(args.plan, args.workdir)
    if args.action == "status":
        for row in plan.status():
            res = row.get("result") or {}
            extra = f" | {res.get('stop_reason', '')} {res.get('steps', '')}" if res else ""
            print(f"{row['id']:<16} {row['type']:<14} {row['status']}{extra}")
    else:
        plan.run(only=args.only, session_max_hours=args.hours)


def cmd_bench(args) -> None:
    from .train.bench import bench
    print(json.dumps(bench(args.config, args.steps, overrides=_parse_overrides(args.set),
                           target_tokens=args.tokens), indent=2))


def cmd_cloud(args) -> None:
    from .cloud.cli import run_cloud_command
    run_cloud_command(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="titan", description="TITAN AI — modelo propio entrenado desde cero")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="preparar y auditar datos")
    d.add_argument("action", choices=["prepare", "audit"])
    d.add_argument("--config")
    d.add_argument("--data-dir")
    d.add_argument("--eval-files", nargs="*", default=["titan/eval/data/*.jsonl"])
    d.set_defaults(func=cmd_data)

    sf = sub.add_parser("sft", help="preparar datos de ajuste de instrucciones")
    sf.add_argument("action", choices=["prepare"])
    sf.add_argument("--config", required=True)
    sf.set_defaults(func=cmd_sft)

    t = sub.add_parser("train", help="preentrenar o ajustar (SFT)")
    t.add_argument("--config", required=True)
    t.add_argument("--set", nargs="*", help="sobrescribir claves: max_steps=100 model.dim=512")
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("generate", help="generar texto")
    g.add_argument("--checkpoint", required=True, help="directorio de ejecución, checkpoint o modelo exportado")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--chat", action="store_true")
    g.add_argument("--device", default="auto")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("eval", help="ejecutar benchmarks")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--suite", default="all")
    e.add_argument("--out")
    e.add_argument("--limit", type=int)
    e.add_argument("--data", help="manifest.json para la suite bpb")
    e.add_argument("--device", default="auto")
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("estimate", help="estimar memoria, tiempo y coste de entrenamiento")
    s.add_argument("--config", help="config de modelo (YAML)")
    s.add_argument("--preset", help="nombre de preset de escalado (ver docs/ESCALADO_Y_PRESUPUESTO.md)")
    s.add_argument("--tokens", type=float, help="tokens de entrenamiento (p. ej. 1e12)")
    s.add_argument("--gpu", default="H100")
    s.add_argument("--num-gpus", type=int, default=8)
    s.add_argument("--price", type=float, help="USD por GPU-hora")
    s.add_argument("--mfu", type=float, default=0.40)
    s.add_argument("--table", action="store_true", help="tabla de todas las escalas")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_estimate)

    x = sub.add_parser("export", help="exportar pesos verificados")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--formats", default="titan,hf,gguf")
    x.set_defaults(func=cmd_export)

    q = sub.add_parser("quantize", help="cuantizar un modelo exportado (int8/int4)")
    q.add_argument("--model", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--bits", type=int, default=8, choices=[4, 8])
    q.add_argument("--group-size", type=int, default=128)
    q.set_defaults(func=cmd_quantize)

    gg = sub.add_parser("gguf", help="cuantizar o medir un modelo GGUF con llama.cpp")
    gg.add_argument("action", choices=["quantize", "bench"])
    gg.add_argument("--model", required=True, help="archivo .gguf")
    gg.add_argument("--out", help="archivo .gguf de salida (quantize)")
    gg.add_argument("--type", default="Q4_K_M", help="Q8_0, Q6_K, Q5_K_M, Q4_K_M, ...")
    gg.add_argument("--prompt-tokens", type=int, default=128)
    gg.add_argument("--gen-tokens", type=int, default=128)
    gg.add_argument("--gpu-layers", type=int, default=-1, help="-1 = todas las capas en GPU")
    gg.set_defaults(func=cmd_gguf)

    v = sub.add_parser("serve", help="servidor HTTP con interfaz web")
    v.add_argument("--model", required=True)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--device", default="auto")
    v.add_argument("--agent-config", default=None)
    v.set_defaults(func=cmd_serve)

    gr = sub.add_parser("grow", help="agrandar un modelo (más capas) o convertirlo en MoE")
    gr.add_argument("op", choices=["depth", "upcycle"])
    gr.add_argument("--source", required=True)
    gr.add_argument("--out", required=True)
    gr.add_argument("--layers", type=int, help="nuevo número de capas (depth)")
    gr.add_argument("--method", default="copy_zero", choices=["copy_zero", "stack"])
    gr.add_argument("--experts", type=int, default=4)
    gr.add_argument("--top-k", type=int, default=2)
    gr.add_argument("--tokenizer")
    gr.set_defaults(func=cmd_grow)

    me = sub.add_parser("merge-experts", help="unir ramas entrenadas por separado en un MoE (BTX)")
    me.add_argument("--branches", nargs="+", required=True)
    me.add_argument("--out", required=True)
    me.add_argument("--top-k", type=int, default=2)
    me.add_argument("--tokenizer")
    me.set_defaults(func=cmd_merge)

    pl = sub.add_parser("plan", help="ejecutar un plan de entrenamiento por fases")
    pl.add_argument("action", choices=["run", "status"])
    pl.add_argument("--plan", required=True)
    pl.add_argument("--workdir")
    pl.add_argument("--only", nargs="*", help="ejecutar solo estas fases")
    pl.add_argument("--hours", type=float, help="límite de horas para esta sesión")
    pl.set_defaults(func=cmd_plan)

    be = sub.add_parser("bench", help="medir tokens/s y memoria en este equipo")
    be.add_argument("--config", required=True)
    be.add_argument("--steps", type=int, default=10)
    be.add_argument("--tokens", type=float, help="tokens objetivo para proyectar horas")
    be.add_argument("--set", nargs="*")
    be.set_defaults(func=cmd_bench)

    c = sub.add_parser("cloud", help="infraestructura GPU en la nube")
    c.add_argument("action", choices=["providers", "offers", "plan", "launch", "status", "stop", "watch", "spend"])
    c.add_argument("--instance", help="ID de la instancia (status/stop/watch)")
    c.add_argument("--job", help="YAML del trabajo")
    c.add_argument("--provider", default="runpod")
    c.add_argument("--gpu", default="H100")
    c.add_argument("--num-gpus", type=int, default=8)
    c.add_argument("--approve", action="store_true", help="autoriza explícitamente el gasto del plan")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(func=cmd_cloud)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
