"""Subcomandos ``titan cloud ...``."""

from __future__ import annotations

import json
import os

from .job import JobSpec, launch, plan, spent_so_far, stop, watch
from .providers.base import ApprovalRequired
from .providers.impl import PROVIDERS, get_provider


def run_cloud_command(args) -> None:
    if args.action == "providers":
        for name, cls in PROVIDERS.items():
            print(f"{name:<8} clave: {cls.api_key_env:<16} {'definida' if os.environ.get(cls.api_key_env) else 'NO definida'}"
                  f"  API: {cls.base_url}")
        return
    if args.action == "offers":
        prov = get_provider(args.provider, dry_run=args.dry_run)
        offers = prov.list_offers(args.gpu, args.num_gpus)
        for o in offers:
            print(f"{o.provider}:{o.offer_id} {o.num_gpus}×{o.gpu} ${o.price_per_gpu_hour:.2f}/GPU-h {o.region}")
        if args.dry_run:
            print(json.dumps(prov.sent, indent=2))
        return
    if args.action == "spend":
        print(json.dumps(spent_so_far(), indent=2))
        return
    spec = JobSpec.load(args.job)
    prov = get_provider(spec.provider, dry_run=args.dry_run)
    if args.action == "plan":
        print(json.dumps(plan(spec, prov), indent=2, ensure_ascii=False))
    elif args.action == "launch":
        p = plan(spec, prov)
        print(json.dumps({k: p[k] for k in ("estimated_hours", "estimated_cost_usd", "budget_usd", "warnings")},
                         indent=2, ensure_ascii=False))
        try:
            inst = launch(spec, approve=args.approve, provider=prov)
        except ApprovalRequired as e:
            raise SystemExit(f"No se lanzó nada: {e}")
        print(f"Instancia creada: {inst.instance_id} ({inst.status}). Vigílela con: titan cloud watch --job {args.job} "
              f"--instance {inst.instance_id}")
    elif args.action == "status":
        print(prov.get(args.instance))
    elif args.action == "stop":
        print(stop(spec, args.instance, prov))
    elif args.action == "watch":
        print(watch(spec, args.instance, prov))
