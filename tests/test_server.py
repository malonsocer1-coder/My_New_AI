import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import torch

from titan.agent.permissions import PermissionRequest
from titan.export.exporter import export_checkpoint
from titan.model import ModelConfig, Transformer
from titan.tokenizer import train_tokenizer
from titan.train import grow
from titan.train.checkpoint import model_state_for_save
from titan.ui.server import Approvals, TitanApp, make_handler


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    d = tmp_path_factory.mktemp("srv")
    tok = train_tokenizer(["Hola, ¿cómo estás? Muy bien, gracias. " * 50] * 10, 300, d / "tok")
    cfg = ModelConfig(vocab_size=300, dim=32, n_layers=1, n_heads=2, max_seq_len=512, multiple_of=16)
    m = Transformer(cfg)
    m.init_weights(seed=1)
    src = grow.save_model_dir(cfg, model_state_for_save(m), d / "src", tokenizer=d / "tok")
    export_checkpoint(src, d / "model", formats=["titan"], dtype="fp32")
    (d / "agent.yaml").write_text(f"root: {d}\nworkspace: {d / 'ws'}\n")
    app = TitanApp(str(d / "model"), device="cpu", agent_config=str(d / "agent.yaml"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield app, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _req(url, data=None, token=None, host=None):
    h = {"Content-Type": "application/json"}
    if token:
        h["X-Titan-Token"] = token
    if host:
        h["Host"] = host
    r = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers=h)
    return urllib.request.urlopen(r, timeout=60)


def test_index_embeds_token_and_security_headers(server):
    app, base = server
    r = _req(base + "/")
    html = r.read().decode()
    assert app.token in html and "DENY" == r.headers["X-Frame-Options"]


def test_api_requires_token_and_local_host(server):
    app, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(base + "/api/info")
    assert e.value.code == 401
    with pytest.raises(urllib.error.HTTPError) as e:  # DNS rebinding: Host ajeno
        _req(base + "/api/info", token=app.token, host="evil.example.com")
    assert e.value.code == 403
    info = json.loads(_req(base + "/api/info", token=app.token).read())
    assert "calculator" in info["tools"] and info["permissions"]["spend_money"] == "ask"


def test_chat_streams_events(server):
    app, base = server
    r = _req(base + "/api/chat", {"messages": [{"role": "user", "content": "Hola"}], "max_new_tokens": 8,
                                  "temperature": 0}, token=app.token)
    events = [json.loads(line[6:]) for line in r.read().decode().split("\n\n") if line.startswith("data: ")]
    assert events[-1]["type"] == "done" and any(e["type"] == "final" for e in events)


def test_approvals_bridge():
    ap = Approvals(timeout_s=5)
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("d", ap.approver(PermissionRequest("code_exec", "python_exec", "x"))))
    t.start()
    for _ in range(100):
        if ap.pending:
            break
        threading.Event().wait(0.02)
    rid = next(iter(ap.pending))
    assert not ap.decide(rid, "hackear")
    assert ap.decide(rid, "allow_once")
    t.join(5)
    assert result["d"] == "allow_once"
