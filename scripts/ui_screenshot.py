"""Arranca el servidor con un modelo, hace una pregunta en modo agente y guarda una captura.

Uso: python scripts/ui_screenshot.py --model models/base/titan-stage-a-sft --out reports/ui_agente.png
"""
import argparse
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.ui.server import TitanApp, make_handler  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="reports/ui_agente.png")
    p.add_argument("--question", default="¿Cuánto es 1234 * 5678?")
    a = p.parse_args()
    tmp = Path(tempfile.mkdtemp())
    # Mismas herramientas que en el SFT: el modelo de la Etapa A tiene un contexto de 768 tokens
    tools = "[calculator, list_dir, read_file, write_file, delete_file, python_exec, kb_search]"
    (tmp / "agent.yaml").write_text(f"root: {tmp}\nworkspace: {tmp / 'ws'}\ntools: {tools}\n", encoding="utf-8")
    app = TitanApp(a.model, device="cpu", agent_config=str(tmp / "agent.yaml"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    js = tmp / "shot.js"
    js.write_text(r"""
const { chromium } = require('playwright');
(async () => {
  const [url, out, question] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1100, height: 760 } });
  await page.goto(url);
  await page.selectOption('#mode', 'agent');
  await page.fill('#temp', '0');
  await page.fill('#input', question);
  await page.click('#send');
  await page.waitForFunction(() => !document.getElementById('send').disabled, null, { timeout: 180000 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: out, fullPage: true });
  const texts = await page.$$eval('.msg', els => els.map(e => e.className + ': ' + e.textContent));
  console.log(texts.join('\n'));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
""", encoding="utf-8")
    import subprocess
    r = subprocess.run(["node", str(js), url, str(Path(a.out).resolve()), a.question], capture_output=True, text=True,
                       timeout=300)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-2000:])
    httpd.shutdown()
    print(f"captura: {a.out}")


if __name__ == "__main__":
    main()
