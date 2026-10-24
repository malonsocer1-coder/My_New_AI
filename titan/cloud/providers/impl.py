"""Clientes de RunPod, Vast.ai y Lambda.

Implementados según la documentación pública de cada API. Las rutas y campos
pueden cambiar: ``titan cloud offers --dry-run`` muestra exactamente qué se
enviaría, y conviene validarlo contra la documentación vigente antes del
primer lanzamiento:
  * RunPod  REST v1   https://rest.runpod.io/v1  (pods)
  * Vast.ai REST v0   https://console.vast.ai/api/v0  (bundles, asks, instances)
  * Lambda  REST v1   https://cloud.lambda.ai/api/v1  (instance-types, instance-operations)
"""

from __future__ import annotations

import json
import urllib.parse

from .base import Instance, Offer, Provider

GPU_NAMES = {
    "runpod": {"H100": "NVIDIA H100 80GB HBM3", "H200": "NVIDIA H200", "B200": "NVIDIA B200",
               "A100": "NVIDIA A100-SXM4-80GB", "RTX4090": "NVIDIA GeForce RTX 4090"},
    "vastai": {"H100": "H100 SXM", "H200": "H200", "B200": "B200", "A100": "A100 SXM4", "RTX4090": "RTX 4090"},
    "lambda": {"H100": "h100_sxm5", "H200": "h200", "B200": "b200_sxm6", "A100": "a100_80gb_sxm4"},
}


class RunPod(Provider):
    name = "runpod"
    api_key_env = "RUNPOD_API_KEY"
    base_url = "https://rest.runpod.io/v1"

    def list_offers(self, gpu: str, num_gpus: int) -> list[Offer]:
        # RunPod no expone un mercado de ofertas por máquina en REST: se usa el tipo de GPU
        # y el precio de referencia configurado o el devuelto por la API de tipos de GPU.
        data = self.request("GET", "/gputypes") if not self.dry_run else []
        out = []
        for t in data or []:
            if t.get("id") == GPU_NAMES["runpod"].get(gpu):
                price = t.get("securePrice") or t.get("communityPrice") or 0
                out.append(Offer("runpod", t["id"], gpu, num_gpus, price * num_gpus, extra=t))
        if not out:
            out.append(Offer("runpod", GPU_NAMES["runpod"].get(gpu, gpu), gpu, num_gpus, 0.0,
                             extra={"note": "precio no disponible: verifique en la consola"}))
        return out

    def create(self, offer: Offer, spec: dict, approved: bool = False) -> Instance:
        self._require(approved)
        body = {
            "name": spec["name"], "imageName": spec["image"], "gpuTypeIds": [offer.offer_id],
            "gpuCount": offer.num_gpus, "containerDiskInGb": spec.get("disk_gb", 200),
            "volumeInGb": spec.get("volume_gb", 0), "cloudType": spec.get("cloud_type", "SECURE"),
            "env": spec.get("env", {}), "ports": spec.get("ports", ["22/tcp"]),
            "dockerStartCmd": ["bash", "-lc", spec["startup_script"]],
        }
        r = self.request("POST", "/pods", body, mutating=True)
        return Instance("runpod", r.get("id", "dry-run"), r.get("desiredStatus", "CREATED"), offer.price_per_hour,
                        extra=r)

    def get(self, instance_id: str) -> Instance:
        r = self.request("GET", f"/pods/{instance_id}")
        return Instance("runpod", instance_id, r.get("desiredStatus", "UNKNOWN"), r.get("costPerHr"), extra=r)

    def terminate(self, instance_id: str) -> dict:
        return self.request("DELETE", f"/pods/{instance_id}", mutating=True)


class VastAI(Provider):
    name = "vastai"
    api_key_env = "VAST_API_KEY"
    base_url = "https://console.vast.ai/api/v0"

    def list_offers(self, gpu: str, num_gpus: int) -> list[Offer]:
        q = {"gpu_name": {"eq": GPU_NAMES["vastai"].get(gpu, gpu)}, "num_gpus": {"eq": num_gpus},
             "rentable": {"eq": True}, "verified": {"eq": True}, "reliability2": {"gte": 0.98},
             "order": [["dph_total", "asc"]], "type": "on-demand"}
        data = self.request("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(q)))
        offers = []
        for o in (data or {}).get("offers", []) if isinstance(data, dict) else []:
            offers.append(Offer("vastai", str(o["id"]), gpu, o.get("num_gpus", num_gpus), float(o.get("dph_total", 0)),
                                region=o.get("geolocation", ""), interconnect=o.get("pci_gen", ""),
                                reliability=o.get("reliability2"), extra={"inet_down": o.get("inet_down")}))
        return offers

    def create(self, offer: Offer, spec: dict, approved: bool = False) -> Instance:
        self._require(approved)
        body = {"client_id": "me", "image": spec["image"], "disk": spec.get("disk_gb", 200),
                "onstart": spec["startup_script"], "env": spec.get("env", {}), "runtype": "ssh"}
        r = self.request("PUT", f"/asks/{offer.offer_id}/", body, mutating=True)
        return Instance("vastai", str(r.get("new_contract", "dry-run")), "CREATED", offer.price_per_hour, extra=r)

    def get(self, instance_id: str) -> Instance:
        r = self.request("GET", f"/instances/{instance_id}/")
        inst = r.get("instances", r) if isinstance(r, dict) else {}
        return Instance("vastai", instance_id, inst.get("actual_status", "UNKNOWN"), inst.get("dph_total"), extra=inst)

    def terminate(self, instance_id: str) -> dict:
        return self.request("DELETE", f"/instances/{instance_id}/", mutating=True)


class Lambda(Provider):
    name = "lambda"
    api_key_env = "LAMBDA_API_KEY"
    base_url = "https://cloud.lambda.ai/api/v1"

    def list_offers(self, gpu: str, num_gpus: int) -> list[Offer]:
        data = self.request("GET", "/instance-types")
        offers = []
        want = GPU_NAMES["lambda"].get(gpu, gpu)
        for name, t in ((data or {}).get("data", {}) if isinstance(data, dict) else {}).items():
            it = t.get("instance_type", {})
            if want in name and it.get("specs", {}).get("gpus") == num_gpus:
                regions = [r["name"] for r in t.get("regions_with_capacity_available", [])]
                offers.append(Offer("lambda", name, gpu, num_gpus, it.get("price_cents_per_hour", 0) / 100,
                                    region=",".join(regions), extra={"available": bool(regions)}))
        return offers

    def create(self, offer: Offer, spec: dict, approved: bool = False) -> Instance:
        self._require(approved)
        body = {"region_name": spec.get("region") or offer.region.split(",")[0], "instance_type_name": offer.offer_id,
                "ssh_key_names": spec.get("ssh_key_names", []), "quantity": 1, "name": spec["name"],
                "user_data": spec["startup_script"]}
        r = self.request("POST", "/instance-operations/launch", body, mutating=True)
        ids = r.get("data", {}).get("instance_ids", ["dry-run"]) if isinstance(r, dict) else ["dry-run"]
        return Instance("lambda", ids[0], "booting", offer.price_per_hour, extra=r)

    def get(self, instance_id: str) -> Instance:
        r = self.request("GET", f"/instances/{instance_id}").get("data", {})
        return Instance("lambda", instance_id, r.get("status", "UNKNOWN"), None, ssh=r.get("ip"), extra=r)

    def terminate(self, instance_id: str) -> dict:
        return self.request("POST", "/instance-operations/terminate", {"instance_ids": [instance_id]}, mutating=True)


PROVIDERS = {"runpod": RunPod, "vastai": VastAI, "lambda": Lambda}


def get_provider(name: str, **kw) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"Proveedor desconocido: {name}. Disponibles: {sorted(PROVIDERS)}")
    return PROVIDERS[name](**kw)
