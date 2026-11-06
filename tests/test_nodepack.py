"""test_nodepack.py - проверка нодпака ComfyUI-PolyStrata (v5) против мок-сервера PolyStrata.

Мок эмулирует реальные эндпоинты (форматы из serve/server.py PolyStrata и docs/VRAM_ELASTIC.md):
  POST /v1/chat/completions, POST /v1/vram, POST /v1/unload, GET /v1/status,
  GET /api/models, POST /api/models/load, POST /v1/load
  + пути ошибок: 400 --vram-elastic, 409 model_busy, 503+Retry-After, частичное освобождение
  (частичный успех = 200, сверять vram_free_mib >= reserve_mib), "not loaded" для /v1/vram,
  404 no model named / model not found, старый сервер без /api/models (фолбэк /v1/load
  работает только при switch_on_request).

Архитектура v6: Chat - SCHEMA V3 (comfy_api.latest) с autogrow-входами img_* (картинки) и
vid_* (видео, каждый пин = отдельное видео кадрами), выходы text+thoughts; reasoning/
temperature/top_p/top_k/seed - ТОЛЬКО в чате; лимиты кадров - в опциональной субноде
Media (max_images/max_video_frames/max_frame_side/video_fps), действуют НА КАЖДЫЙ
пин своего типа (0 = все кадры пина, потолки 64/128); кадры заранее подгоняются под
контекст движка (/v1/status -> context.native), потому что двигок запросы не обрезает;
reasoning=none -> reasoning_effort=none + chat_template_kwargs enable_thinking=false
(оба механизма, которые сервер читает). INTERRUPT (Cancel ComfyUI):
throw_exception_if_processing_interrupted поллится во время запроса (interruptible-путь
на http.client: при срабатывании СОКЕТ закрывается - FIN - watch-поток PolyStrata
(_watch_client) видит EOF и отменяет генерацию; Session.close() requests сокет
висящего запроса не закрывал - сервер дорабатывал до конца, исправлено в 0.7.1);
InterruptProcessingException наследует BaseException.
Sampling - без temperature/top_p/top_k/reasoning/seed (они в чате), + logit_bias/
extra_sampling; Media - опц. субнода лимитов кадров; Memory; Unload (полная выгрузка +
сброс серверного резерва + возврат EXTRA_RESERVED_VRAM ComfyUI, сквозная). СМЕНА МОДЕЛИ
(v0.9.0): виджет model в Chat (V3+V1) - дропдаун установленных моделей сервера
(GET /api/models; сентинел 'как на сервере' первым, unavailable скрыты, кеш 15 c -
обновляется F5/R через /object_info); выбранная модель, если не текущая, ПЕРЕКЛЮЧАЕТ
сервер ДО генерации (POST /api/models/load; текущая/сентинел - поле 'model' в chat-запрос
НЕ пишется). Виджетов api_key/extra_mib нет: доступ localhost, резерв - один reserve_mib.
Бронь VRAM для ComfyUI - метод llama.cpp-нод: фактическое потребление
PolyStrata (status.machine.gpu.total_mib - vram_free_mib) -> EXTRA_RESERVED_VRAM.
АВТО-ВЫГРУЗКА (механика llama.cpp-нод): Chat помечает сервер как используемый; хук на
PromptExecutor.execute выгружает сервер-сироту (главная нода забайпашена - новый фронт шлёт
BypassedNode__, старый mode=4 / открыт другой воркфлоу / ноды удалены); хук на
model_management.unload_all_models выгружает всё наблюдаемое вне выполнения нод. Сервер,
к которому нодпак ни разу не обращался, не трогается (им может пользоваться Jan/curl).

Запуск: python3 tests/test_nodepack.py   (или scripts/test_nodepack.py из workspace)
"""
import base64
import contextlib
import importlib.util
import io
import json
import os
import sys
import threading
import types
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAND1 = os.path.normpath(os.path.join(_HERE, "..", "download", "ComfyUI-PolyStrata"))
_CAND2 = os.path.normpath(os.path.join(_HERE, ".."))
PACK_DIR = os.environ.get("PS_PACK_DIR") or (_CAND1 if os.path.isfile(os.path.join(_CAND1, "nodes.py")) else _CAND2)

# ------------------------------------------------------------------ стаб comfy (schema V3 + model_management)
def install_comfy_stubs():
    """comfy_api.latest.io (schema V3) и comfy.model_management - минимальные заглушки."""
    class _Pin:
        def __init__(self, kind, name, **kw):
            self.kind, self.name, self.kw = kind, name, kw

    def _make(kind):
        def _input(cls, name, **kw):
            return _Pin(kind, name, **kw)

        def _output(cls, display_name=None, **kw):
            return _Pin(kind + "_out", display_name or "", **kw)

        return type(f"_T{kind}", (_Pin,), {"Input": classmethod(_input), "Output": classmethod(_output)})

    class _Autogrow:
        class TemplatePrefix:
            def __init__(self, pin, prefix="x", min=0, max=50):
                self.pin, self.prefix, self.min, self.max = pin, prefix, min, max

        @classmethod
        def Input(cls, name, template=None, **kw):
            return _Pin("autogrow", name, template=template, **kw)

    class _CustomNS:
        def __init__(self, type_id):
            self.type_id = type_id

        def Input(self, name, **kw):
            return _Pin("custom", name, type_id=self.type_id, **kw)

        def Output(self, display_name=None, **kw):
            return _Pin("custom_out", display_name or "", type_id=self.type_id, **kw)

    io_stub = types.ModuleType("comfy_api.latest.io")
    io_stub.ComfyNode = type("ComfyNode", (), {})
    io_stub.Schema = staticmethod(lambda **kw: kw)
    io_stub.Image, io_stub.String, io_stub.Int = _make("IMAGE"), _make("STRING"), _make("INT")
    io_stub.Float, io_stub.Combo, io_stub.Boolean = _make("FLOAT"), _make("COMBO"), _make("BOOLEAN")
    io_stub.Autogrow = _Autogrow
    io_stub.Custom = lambda type_id: _CustomNS(type_id)
    io_stub.NodeOutput = staticmethod(lambda *vals, **kw: ("NodeOutput", vals))

    latest = types.ModuleType("comfy_api.latest")
    latest.io = io_stub
    latest.ComfyExtension = type("ComfyExtension", (), {})
    api = types.ModuleType("comfy_api")
    api.latest = latest
    sys.modules["comfy_api"] = api
    sys.modules["comfy_api.latest"] = latest
    sys.modules["comfy_api.latest.io"] = io_stub

    # folder_paths: models_dir во временной папке (нужен для пресетов системных
    # промптов models/LLM/prompts - тот же путь, что в llama.cpp-нодах).
    import tempfile
    fp = sys.modules.get("folder_paths")
    if fp is None:
        fp = types.ModuleType("folder_paths")
        fp.models_dir = tempfile.mkdtemp(prefix="ps_test_models_")
        fp.folder_names_and_paths = {}
        sys.modules["folder_paths"] = fp

    mm = sys.modules.get("comfy.model_management")
    if mm is None:
        mm = types.ModuleType("comfy.model_management")
        mm.EXTRA_RESERVED_VRAM = 600 * 1024 * 1024      # как в ядре ComfyUI (Windows/16GB+)
        comfy = types.ModuleType("comfy")
        comfy.model_management = mm
        comfy.memory_management = types.ModuleType("comfy.memory_management")
        sys.modules["comfy"] = comfy
        sys.modules["comfy.model_management"] = mm
        sys.modules["comfy.memory_management"] = comfy.memory_management
    # interrupt API (кнопка Cancel) - как в реальном comfy.model_management (master):
    # InterruptProcessingException наследует BaseException, throw_... сбрасывает флаг при срабатывании.
    mm.InterruptProcessingException = type("InterruptProcessingException", (BaseException,), {})
    mm._interrupt_flag = {"on": False}

    def _interrupt_current_processing(value=True):
        mm._interrupt_flag["on"] = bool(value)

    def _throw_exception_if_processing_interrupted():
        if mm._interrupt_flag["on"]:
            mm._interrupt_flag["on"] = False
            raise mm.InterruptProcessingException("processing interrupted")

    mm.interrupt_current_processing = _interrupt_current_processing
    mm.throw_exception_if_processing_interrupted = _throw_exception_if_processing_interrupted

    # очистка памяти ComfyUI (тоггл clear_comfy_vram): счётчики вызовов unload/кэша
    mm.unload_calls = {"n": 0}
    mm.cache_calls = {"n": 0}

    def _stub_unload_all_models():
        mm.unload_calls["n"] += 1

    def _stub_soft_empty_cache():
        mm.cache_calls["n"] += 1

    mm.unload_all_models = _stub_unload_all_models
    mm.soft_empty_cache = _stub_soft_empty_cache
    return io_stub, mm


IO_STUB, MM_STUB = install_comfy_stubs()


def block_comfy_api():
    """Сделать 'import comfy_api.latest' падающим (проверка V1-фолбэка). Возвращает заглушки на возврат."""
    saved = tuple((k, sys.modules.pop(k)) for k in
                  ("comfy_api", "comfy_api.latest", "comfy_api.latest.io")
                  if k in sys.modules)
    sys.modules["comfy_api"] = None          # import поднимет ImportError
    return saved


def restore_comfy_api(saved):
    sys.modules.pop("comfy_api", None)
    for k, v in saved:
        sys.modules[k] = v


# ------------------------------------------------------------------ мок-сервер
class MockState:
    def __init__(self):
        self.elastic = True
        self.last_chat = None          # разобранный JSON последнего chat-запроса
        self.last_vram = None
        self.last_status = False       # запрашивался ли /v1/status
        self.sleeping = False
        self.reserve = None
        self.cache_mib = 6000
        self.free_mib = 700
        self.thoughts = "МОК-МЫСЛИ: смотрю на кадры."   # reasoning_content в ответе
        self.ctx_n = 32768             # окно движка (/v1/status context.native)
        self.props_ctx = 0             # n_ctx в /props (запасной путь; 0 - берётся из status)
        self.ctx_overflow = False      # True -> chat отвечает 400 exceeds the context window
        self.started = 1700000000      # время старта движка (ключ кэша контекста)
        self.chat_delay = 0.0          # сек «генерации» перед ответом (тест interrupt)
        self.disconnect_seen = False   # сервер заметил обрыв клиента (EOF) во время генерации
        self.disconnect_dt = None      # через сколько секунд после начала «генерации»
        # ---- реестр моделей (GET /api/models, POST /api/models/load) - serve/models.py
        self.models = [
            {"id": "gemma-4-26b-a4b", "config": "strata-gemma-4-26b-a4b.json",
             "display_name": "Gemma 4 26B", "backend": "gemma4", "quantization": "Q4_K_M",
             "params_total": "26B", "params_active": "4B", "context": 32768,
             "multimodal": True, "aliases": ["gemma4"], "mtp": False, "ram_gb": 15.0,
             "vram_gb": {"least": 6.0, "all": 16.0}, "status": "loaded", "problem": None},
            {"id": "sage-a1.5-ablitrex-abliterated-bf16.i1-q4_k_m",
             "config": "strata-sage-a1.5.json", "display_name": "Sage A1.5",
             "backend": "qwen4exp", "quantization": "i1-q4_k_m", "params_total": None,
             "params_active": None, "context": 128000, "multimodal": False, "aliases": [],
             "mtp": True, "ram_gb": 9.0, "vram_gb": {"least": 3.0, "all": 10.0},
             "status": "available", "problem": None},
            {"id": "qwen3.8-flash-next-iq3_s", "config": "strata-qwen3.8.json",
             "display_name": "Qwen3.8 Flash Next", "backend": "qwen4exp",
             "quantization": "iq3_s", "params_total": None, "params_active": None,
             "context": 131072, "multimodal": True, "aliases": ["qwen-flash"], "mtp": False,
             "ram_gb": 5.0, "vram_gb": {"least": 2.0, "all": 6.0},
             "status": "available", "problem": None},
            {"id": "broken-model", "config": "strata-broken.json",
             "display_name": "Broken", "backend": "unknown", "quantization": None,
             "params_total": None, "params_active": None, "context": None,
             "multimodal": False, "aliases": [], "mtp": False, "ram_gb": None,
             "vram_gb": None, "status": "unavailable",
             "problem": "the model file is missing"},
        ]
        self.current = "gemma-4-26b-a4b"   # загруженная модель (id)
        self.load_calls = []                 # [{"path": ..., "model": ...}] - переключения
        self.old_server = False              # True -> 404 на /api/models и /api/models/load
        self.load_fallback_ok = False        # /v1/load принимает ЧУЖОЕ имя (switch_on_request)


ST = MockState()
TOTAL_GPU_MIB = 32608            # machine.gpu.total_mib в /v1/status


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):       # тишина в логах
        pass

    def _json(self, code, body, headers=None):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        if self.path == "/v1/status":
            ST.last_status = True
            self._json(200, {
                "service": "strata", "model": ST.current, "loaded": not ST.sleeping,
                "engine": "strata-poly 0.1.45-poly.6", "uptime_s": 1234, "started": ST.started,
                "cache_max_tokens": ST.ctx_n,
                "context": {"native": ST.ctx_n, "max_positions": ST.ctx_n},
                "vram": {"reserve_mib": ST.reserve, "expert_slots": 1200, "expert_slots_full": 3700,
                         "expert_cache_mib": ST.cache_mib, "expert_cache_full_mib": 6000,
                         "vram_free_mib": ST.free_mib, "elastic": True},
                "vision": {"enabled": True, "available": True, "error": None},
                "activity": {"requests": 7, "in_flight": 0, "last_request_at": 1700000000},
                "machine": {"gpu": {"name": "RTX 5090", "used_mib": TOTAL_GPU_MIB - ST.free_mib,
                                    "total_mib": TOTAL_GPU_MIB, "util_pct": 3, "temp_c": 41}},
            })
            return
        if self.path == "/api/models":
            if ST.old_server:                 # сервер до появления реестра в API
                self._json(404, {"error": {"message": "not found"}})
                return
            models = []
            for e in ST.models:
                e2 = dict(e)
                e2["current"] = e2["id"] == ST.current
                if e2["current"]:
                    e2["status"] = "loaded" if not ST.sleeping else "unloaded"
                models.append(e2)
            self._json(200, {"models": models, "current": ST.current})
            return
        if self.path == "/props":
            self._json(200, {"default_generation_settings": {"n_ctx": ST.props_ctx or ST.ctx_n},
                             "model_alias": "gemma-4-26b-a4b"})
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        parsed = urllib.parse.urlsplit(self.path)
        path, query = parsed.path, urllib.parse.parse_qs(parsed.query)
        if path == "/v1/chat/completions":
            req = self._body()
            ST.last_chat = req
            if ST.chat_delay:                 # эмуляция долгой генерации (тест interrupt)
                import select as _sel
                import socket as _sock
                import time as _t
                t0 = _t.monotonic()
                end = t0 + ST.chat_delay
                while _t.monotonic() < end:
                    # как _watch_client PolyStrata (select + MSG_PEEK): EOF = клиент повесил трубку
                    try:
                        r, _, _ = _sel.select([self.connection], [], [], 0)
                        gone = bool(r) and self.connection.recv(1, _sock.MSG_PEEK) == b""
                    except OSError:
                        gone = True
                    if gone:
                        ST.disconnect_seen = True
                        ST.disconnect_dt = _t.monotonic() - t0
                        return                # клиента нет - ответ не пишем
                    _t.sleep(0.05)
            if ST.ctx_overflow:
                self._json(400, {"error": {"message": "invalid request: request exceeds the context "
                                                       "window: prompt (118046 tokens) leaves no room "
                                                       "to answer in the context (32768); requests "
                                                       "are never truncated"}})
                return
            n_imgs = sum(1 for m in req.get("messages", []) if isinstance(m.get("content"), list)
                         for p in m["content"] if p.get("type") == "image_url")
            self._json(200, {
                "choices": [{"message": {"role": "assistant", "content": "МОК-ОТВЕТ: всё вижу.",
                                         "reasoning_content": ST.thoughts},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100 + 300 * n_imgs, "completion_tokens": 9},
                "model": "gemma-4-26b-a4b"})
            return
        if path == "/v1/vram":
            req = self._body()
            ST.last_vram = req
            r = req.get("reserve_mib")
            if not ST.elastic:
                self._json(400, {"error": {"message": "VRAM needs an engine started with "
                                                      "--vram-elastic (one NVIDIA GPU, --serve)"}})
                return
            if ST.sleeping:
                # на спящей модели: ничего на GPU, только записать резерв на следующую загрузку
                ST.reserve = r
                self._json(200, {"status": "not loaded", "reserve_mib": r,
                                 "note": "applied when the model loads"})
                return
            if r is None:
                ST.reserve, ST.cache_mib, ST.free_mib = None, 6000, 700
                self._json(200, {"status": "ok", "reserve_mib": None, "expert_slots": 3700,
                                 "expert_slots_full": 3700, "expert_cache_mib": 6000,
                                 "expert_cache_full_mib": 6000, "vram_free_mib": 700,
                                 "prompt_chunk": 4096})
                return
            ST.reserve = r
            freed = min(r, 4000)                     # > 4000 MiB экспертов нет -> частичный успех
            ST.cache_mib, ST.free_mib = 6000 - freed, freed + 700
            self._json(200, {"status": "ok", "reserve_mib": r, "expert_slots": max(100, 4000 - freed),
                             "expert_slots_full": 3700, "expert_cache_mib": ST.cache_mib,
                             "expert_cache_full_mib": 6000, "vram_free_mib": ST.free_mib,
                             "prompt_chunk": 4096})
            return
        if path == "/api/models/load":
            req = self._body()
            ST.load_calls.append({"path": "/api/models/load", "model": req.get("model")})
            if ST.old_server:
                self._json(404, {"error": {"message": "not found"}})
                return
            name = req.get("model")
            entry = next((e for e in ST.models if e["id"] == name), None)
            if entry is None:
                self._json(404, {"error": {"message": f"no model named {name!r}",
                                           "models": [e["id"] for e in ST.models]}})
                return
            ST.current, ST.sleeping = entry["id"], False
            self._json(200, {"status": "loaded",
                             "model": {**entry, "status": "loaded", "current": True}})
            return
        if path == "/v1/load":
            req = self._body()
            ST.load_calls.append({"path": "/v1/load", "model": req.get("model")})
            m = req.get("model")
            if m not in (None, ST.current):
                if not ST.load_fallback_ok:   # switch_on_request выключен (#38)
                    self._json(404, {"error": {"message": "model not found"}})
                    return
                entry = next((e for e in ST.models if e["id"] == m), None)
                if entry is not None:
                    ST.current, ST.sleeping = entry["id"], False
            self._json(200, {"status": "loaded", "model": ST.current})
            return
        if path == "/v1/unload":
            wait = query.get("wait", [None])[0]
            if wait in (None, "0", "false"):
                self._json(409, {"error": {"type": "model_busy",
                                           "message": "a request is running or queued"}})
                return
            ST.sleeping, ST.reserve, ST.cache_mib, ST.free_mib = True, None, 0, TOTAL_GPU_MIB
            self._json(200, {"status": "unloaded"})
            return
        self._json(404, {"error": {"message": "not found"}})


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass          # обрыв клиента (interrupt) - норма, трейсбеки в stderr не нужны


def start_mock():
    srv = _QuietServer(("127.0.0.1", 0), MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"127.0.0.1:{srv.server_address[1]}"


def load_pack(as_package: bool):
    """as_package=True - как современный ComfyUI (package-спека); False - как старый загрузчик."""
    name = "ComfyUI-PolyStrata" if as_package else "PolyStrataPlain"
    if as_package:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(PACK_DIR, "__init__.py"), submodule_search_locations=[PACK_DIR])
    else:
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, os.path.join(PACK_DIR, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_fake_image_batch(n=2, h=48, w=72):
    rng = np.random.default_rng(3)
    return rng.random((n, h, w, 3), dtype=np.float32)


def make_media(max_frame_side=1024, max_images=16, max_video_frames=8, video_fps=0):
    """Субнода PolyStrataMedia с заданными лимитами (как после исполнения ноды).
    Модуль берётся из sys.modules: в V1-секции пак перезагружен без comfy_api -
    ленивый import вернёт актуальный экземпляр (PolyStrataMedia в обеих ветках общий)."""
    import importlib as _il
    nodes = _il.import_module("ComfyUI-PolyStrata.nodes")
    m = nodes.PolyStrataMedia()
    m.run(max_frame_side=max_frame_side, max_images=max_images,
          max_video_frames=max_video_frames, video_fps=video_fps)
    return m


# ------------------------------------------------------------------ проверки
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)


def capture(fn, *a, **kw):
    """Запуск с перехватом stdout (проверяем лог-строки нод)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


def media_parts(req):
    """image_url-части последнего сообщения chat-запроса (пусто, если content - строка)."""
    m = req["messages"][-1]
    if not isinstance(m.get("content"), list):
        return []
    return [p for p in m["content"] if p.get("type") == "image_url"]


def text_parts(req):
    m = req["messages"][-1]
    return [p.get("text", "") for p in m["content"] if p.get("type") == "text"]


def main():
    server = start_mock()
    print(f"Мок-сервер PolyStrata: {server} (пак: {PACK_DIR})")

    # ---------- 1. импорт нодпака двумя способами ----------
    print("\n[1] Импорт пакета")
    pack = load_pack(as_package=True)
    expected = {"PolyStrataChat", "PolyStrataSampling", "PolyStrataMedia",
                "PolyStrataMemory", "PolyStrataUnload"}
    check("NODE_CLASS_MAPPINGS (package-загрузка)", set(pack.NODE_CLASS_MAPPINGS) == expected,
          str(sorted(pack.NODE_CLASS_MAPPINGS)))
    import importlib as _il
    nodes_mod = _il.import_module("ComfyUI-PolyStrata.nodes")
    check("schema V3 обнаружена (comfy_api.latest)", nodes_mod._HAS_V3 is True)
    check("Chat - V3-класс (define_schema)", hasattr(nodes_mod.PolyStrataChat, "define_schema"))
    pack2 = load_pack(as_package=False)
    check("NODE_CLASS_MAPPINGS (старый загрузчик)", set(pack2.NODE_CLASS_MAPPINGS) == expected)
    check("нод Load/VRAM больше нет",
          not any(k for k in pack.NODE_CLASS_MAPPINGS if "Load" in k or "Vram" in k))
    check("версия 0.9.0", getattr(pack, "__version__", "") == "0.9.0")

    samp_node = nodes_mod.PolyStrataSampling()
    mem_node = nodes_mod.PolyStrataMemory()
    unload_node = nodes_mod.PolyStrataUnload()

    # ---------- 2. Schema V3 ноды Chat ----------
    print("\n[2] Schema V3: autogrow-входы картинок и видео отдельно")
    nodes_mod.mark_server_active(server)   # дропдаун model строится по серверам сессии
    nodes_mod._MODEL_LIST_CACHE.clear()
    schema = nodes_mod.PolyStrataChat.define_schema()
    ins = {i.name: i for i in schema["inputs"]}
    outs = schema["outputs"]
    check("node_id/display_name", schema["node_id"] == "PolyStrataChat"
          and schema["display_name"] == "PolyStrata Chat")
    ag = [i for i in schema["inputs"] if i.kind == "autogrow"]
    check("ровно два autogrow-входа: images и videos", [i.name for i in ag] == ["images", "videos"],
          str([i.name for i in ag]))
    tpl = {i.name: i.kw["template"] for i in ag}
    check("префиксы РАЗНЫЕ: img_* / vid_*",
          tpl["images"].prefix == "img" and tpl["videos"].prefix == "vid")
    check("autogrow до 50 пинов, min=0",
          tpl["images"].max == 50 and tpl["videos"].max == 50
          and tpl["images"].min == 0 and tpl["videos"].min == 0)
    check("видео-пины - тип IMAGE (батч кадров), не VIDEO", tpl["videos"].pin.kind == "IMAGE")
    check("memory/sampling/media - кастомные типы PS_*",
          ins["memory"].kw.get("type_id") == "PS_MEMORY"
          and ins["sampling"].kw.get("type_id") == "PS_SAMPLING"
          and ins["media"].kw.get("type_id") == "PS_MEDIA")
    check("выходы text + thoughts",
          [o.name for o in outs] == ["text", "thoughts"])
    check("виджеты: reasoning/temperature/top_p в ГЛАВНОЙ ноде",
          all(n in ins for n in ("reasoning", "temperature", "top_p")))
    check("виджеты: reserve_mib и clear_comfy_vram в ГЛАВНОЙ ноде",
          "reserve_mib" in ins and "clear_comfy_vram" in ins)
    check("виджет top_k - в чате ПОСЛЕ top_p (переехал из Sampling)",
          "top_k" in ins and list(ins).index("top_p") < list(ins).index("top_k"))
    check("виджет seed - тоже в ГЛАВНОЙ ноде", "seed" in ins)
    check("виджеты server/prompt/system_prompt_preset/system_prompt_text",
          all(n in ins for n in ("server", "prompt", "system_prompt_preset",
                                 "system_prompt_text")))
    check("мусор v0.7 вычищен: НЕТ api_key/extra_mib и медиа-лимитов в чате (model - ЕСТЬ, v0.9.0)",
          not ({"api_key", "extra_mib", "max_frame_side", "max_images",
                "max_video_frames", "video_fps"} & set(ins)) and "model" in ins)
    check("виджет model СРАЗУ после server (v0.9.0)",
          "model" in ins and list(ins).index("server") < list(ins).index("model")
          and ins["model"].kind == "COMBO")
    check("model: сентинел 'как на сервере' первым и он же default",
          ins["model"].kw["options"][0] == nodes_mod.MODEL_SWITCH_OFF
          and ins["model"].kw["default"] == nodes_mod.MODEL_SWITCH_OFF)
    check("model: дропдаун - модели сервера, unavailable скрыт",
          {"gemma-4-26b-a4b", "sage-a1.5-ablitrex-abliterated-bf16.i1-q4_k_m",
           "qwen3.8-flash-next-iq3_s"} <= set(ins["model"].kw["options"])
          and "broken-model" not in ins["model"].kw["options"],
          str(ins["model"].kw["options"]))
    check("пресет: список опций начинается с 'none' и содержит .txt из models/LLM/prompts",
          ins["system_prompt_preset"].kw["options"][0] == nodes_mod.NO_SYSTEM_PROMPT
          and isinstance(ins["system_prompt_preset"].kw["options"], list))

    # ---------- 3. Sampling: без дубликатов + logit_bias/extra_sampling ----------
    print("\n[3] Sampling: только доп. сэмплинг, temperature/top_p/reasoning НЕ дублируются")
    s = samp_node.INPUT_TYPES()["required"]
    check("в Sampling НЕТ temperature/top_p/reasoning",
          not ({"temperature", "top_p", "reasoning"} & set(s)), str(sorted(s)))
    check("в Sampling НЕТ top_k (переехал в чат)", "top_k" not in s, str(sorted(s)))
    check("в Sampling есть min_p/штрафы/stop",
          {"min_p", "repeat_penalty", "repeat_last_n", "presence_penalty",
           "frequency_penalty", "stop"} <= set(s))
    check("в Sampling НЕТ seed (он переехал в ноду Chat)", "seed" not in s, str(sorted(s)))
    check("в Sampling есть logit_bias и extra_sampling",
          {"logit_bias", "extra_sampling"} <= set(s))
    tt = json.dumps(s, ensure_ascii=False)
    check("тултип объясняет, где xtc/dry/mirostat (SAMPLERS_NOTE)",
          "xtc" in tt and "extra_sampling" in tt)
    check("тултипы НЕ просят добавлять vram_elastic", "добавьте" not in tt)

    out, log = capture(samp_node.run, min_p=0.05, repeat_penalty=1.1, repeat_last_n=64,
                       presence_penalty=0.3, frequency_penalty=0.7, stop="КОНЕЦ\nДОВОЛЬНО\n",
                       logit_bias="12345\n678:-2.5\n# комментарий\n999 = 1.5",
                       extra_sampling='{"xtc_probability": 0.5}')
    smp = out[0]
    check("заданные поля в словаре, sentinel -1 выброшен",
          smp.get("top_k") is None and smp.get("min_p") == 0.05 and smp.get("repeat_penalty") == 1.1
          and smp.get("repeat_last_n") == 64 and smp.get("presence_penalty") == 0.3
          and smp.get("frequency_penalty") == 0.7 and smp.get("stop") == ["КОНЕЦ", "ДОВОЛЬНО"],
          str(smp))
    check("logit_bias разобран (бан = -100)",
          smp.get("logit_bias") == {"12345": -100.0, "678": -2.5, "999": 1.5}, str(smp.get("logit_bias")))
    check("extra_sampling разобран", smp.get("extra") == {"xtc_probability": 0.5})
    out, _ = capture(samp_node.run)   # все сантинелы
    check("все сантинелы -> пустой словарь", out[0] == {}, str(out[0]))
    try:
        samp_node.run(logit_bias="мусор")
        check("logit_bias: мусор -> ошибка", False)
    except nodes_mod.PolyStrataError:
        check("logit_bias: мусор -> ошибка", True)
    try:
        samp_node.run(extra_sampling="[1,2]")
        check("extra_sampling: не объект -> ошибка", False)
    except nodes_mod.PolyStrataError:
        check("extra_sampling: не объект -> ошибка", True)

    # ---------- 3b. Media: опциональная субнода медиа-лимитов (без сентинелов) ----------
    print("\n[3b] Media: субнода-оверрайд лимитов кадров для Chat")
    mi = nodes_mod.PolyStrataMedia.INPUT_TYPES()["required"]
    check("Media: ровно 4 виджета",
          set(mi) == {"max_frame_side", "max_images", "max_video_frames", "video_fps"},
          str(sorted(mi)))
    check("Media: дефолты = реальные значения (1024/16/8/0), без сентинелов",
          mi["max_frame_side"][1]["default"] == 1024 and mi["max_images"][1]["default"] == 16
          and mi["max_video_frames"][1]["default"] == 8 and mi["video_fps"][1]["default"] == 0)
    media_node = nodes_mod.PolyStrataMedia()
    out, log = capture(media_node.run, max_frame_side=512, max_images=0, max_video_frames=3,
                       video_fps=2)
    md = out[0]
    check("Media run -> объект с атрибутами",
          (md.max_frame_side, md.max_images, md.max_video_frames, md.video_fps)
          == (512, 0, 3, 2), str(vars(md)))

    # ---------- 4. Chat V3: одиночный запрос ----------
    print("\n[4] Chat V3 - одиночный запрос (execute)")
    ST.sleeping = False
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="Привет!",
                       system_prompt_preset=nodes_mod.NO_SYSTEM_PROMPT,
                       system_prompt_text="Ты краток.", max_tokens=256, temperature=0.5, top_p=0.9)
    check("NodeOutput(text, thoughts)", out == ("NodeOutput", ("МОК-ОТВЕТ: всё вижу.", ST.thoughts)),
          str(out))
    req = ST.last_chat
    check("stream=False", req.get("stream") is False)
    check("max_tokens передан", req.get("max_tokens") == 256)
    check("поле 'model' НЕ отправляется (модель - запущенная на сервере)", "model" not in req)
    check("system+user сообщения", [m["role"] for m in req["messages"]] == ["system", "user"])
    check("content строкой (без медиа)", isinstance(req["messages"][-1]["content"], str))
    check("temperature/top_p от виджетов", req.get("temperature") == 0.5 and req.get("top_p") == 0.9)
    check("top_k/seed/stop/logit_bias не отправляются без Sampling",
          "top_k" not in req and "seed" not in req and "stop" not in req and "logit_bias" not in req)
    check("reserve_mib=0: /v1/vram НЕ вызывается", ST.last_vram is None)
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="x", reasoning="none")
    check("reasoning=none -> reasoning_effort=none И chat_template_kwargs (оба механизма)",
          ST.last_chat.get("reasoning_effort") == "none"
          and ST.last_chat.get("chat_template_kwargs") == {"enable_thinking": False}
          and "enable_thinking" not in ST.last_chat,
          str({k: ST.last_chat.get(k) for k in ("reasoning_effort", "chat_template_kwargs",
                                                "enable_thinking")}))
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="x", reasoning="high")
    check("reasoning=high -> reasoning_effort=high", ST.last_chat.get("reasoning_effort") == "high")
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="x", seed=4242)
    check("seed из виджета чата уходит в запрос", ST.last_chat.get("seed") == 4242)
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="x")
    check("seed=0 (случайный) не отправляется", "seed" not in ST.last_chat)
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="x", top_k=33)
    check("top_k из виджета чата уходит в запрос", ST.last_chat.get("top_k") == 33)

    # ---------- 4b. Смена модели (v0.9.0): виджет model в главной ноде ----------
    print("\n[4b] Смена модели: дропдаун, сентинел, переключение, ошибки")
    nodes_mod._MODEL_LIST_CACHE.clear()
    opts = nodes_mod._model_combo_options()
    check("_model_combo_options: сентинел первым + модели сессии (unavailable скрыт)",
          opts[0] == nodes_mod.MODEL_SWITCH_OFF
          and {"gemma-4-26b-a4b", "sage-a1.5-ablitrex-abliterated-bf16.i1-q4_k_m",
               "qwen3.8-flash-next-iq3_s"} <= set(opts)
          and "broken-model" not in opts, str(opts))
    ids = nodes_mod._server_model_ids(server)
    check("_server_model_ids: только id, unavailable скрыт",
          ids == ["gemma-4-26b-a4b", "sage-a1.5-ablitrex-abliterated-bf16.i1-q4_k_m",
                  "qwen3.8-flash-next-iq3_s"], str(ids))
    ST.load_calls.clear()
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="сентинел",
                       model=nodes_mod.MODEL_SWITCH_OFF)
    check("сентинел 'как на сервере': сервер НЕ трогается (нет load-вызовов)",
          not ST.load_calls and ST.current == "gemma-4-26b-a4b")
    check("сентинел: поле 'model' в chat-запрос НЕ пишется", "model" not in ST.last_chat)
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="текущая",
                       model="gemma-4-26b-a4b")
    check("текущая модель: load НЕ вызывается (лишний раз сервер не дёргаем)",
          not ST.load_calls and "model" not in ST.last_chat)
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server,
                       prompt="переключись", model="qwen3.8-flash-next-iq3_s")
    check("чужая модель: POST /api/models/load {model: id}",
          ST.load_calls and ST.load_calls[-1] == {"path": "/api/models/load",
                                                  "model": "qwen3.8-flash-next-iq3_s"},
          str(ST.load_calls))
    check("сервер переключился (current в моке сменился)", ST.current == "qwen3.8-flash-next-iq3_s")
    check("chat-запрос идёт НОВОЙ модели БЕЗ поля 'model'", "model" not in ST.last_chat)
    check("в логе - 'смена модели'", "смена модели" in log, log[-400:])
    ST.current = "gemma-4-26b-a4b"                  # вернуть для остальных тестов
    try:
        capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="?",
                model="no-such-model")
        check("404 no model named -> понятная ошибка", False)
    except nodes_mod.PolyStrataError as e:
        check("404 no model named -> понятная ошибка (сервер не знает такую модель)",
              "не знает такую модель" in str(e), str(e)[:200])
    # старый сервер: реестра нет - дропдаун из /v1/status, переключение через /v1/load
    nodes_mod._MODEL_LIST_CACHE.clear()
    ST.old_server = True
    ids = nodes_mod._server_model_ids(server)
    check("старый сервер (404 /api/models): список = текущая модель из /v1/status",
          ids == ["gemma-4-26b-a4b"], str(ids))
    try:
        capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="?",
                model="qwen3.8-flash-next-iq3_s")
        check("старый сервер без switch_on_request: /v1/load -> 404 model not found", False)
    except nodes_mod.PolyStrataError as e:
        check("старый сервер без switch_on_request: /v1/load -> 404 model not found",
              "model not found" in str(e), str(e)[:200])
        check("фолбэк дошёл до /v1/load",
              ST.load_calls and ST.load_calls[-1]["path"] == "/v1/load", str(ST.load_calls))
    ST.load_fallback_ok = True                       # switch_on_request включён
    ST.current = "gemma-4-26b-a4b"
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="фолбэк",
                       model="qwen3.8-flash-next-iq3_s")
    check("старый сервер + switch_on_request: /v1/load переключил и генерация прошла",
          ST.load_calls[-1] == {"path": "/v1/load", "model": "qwen3.8-flash-next-iq3_s"}
          and ST.current == "qwen3.8-flash-next-iq3_s"
          and out[1][0] == "МОК-ОТВЕТ: всё вижу.", str(ST.load_calls[-2:]))
    ST.old_server = False
    ST.load_fallback_ok = False
    ST.current = "gemma-4-26b-a4b"
    nodes_mod._MODEL_LIST_CACHE.clear()

    # ---------- 5. Chat V3: картинки и видео РАЗДЕЛЬНО ----------
    print("\n[5] Chat V3 - autogrow: img_* и vid_* отдельно")
    img_a = make_fake_image_batch(2)          # img_0: 2 кадра
    img_b = make_fake_image_batch(1)          # img_1: 1 кадр
    vid_a = make_fake_image_batch(8)          # vid_0: 8 кадров
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="Опиши.",
                     images={"img_0": img_a, "img_1": img_b}, videos={"vid_0": vid_a},
                     media=make_media(max_images=0, max_video_frames=4))
    req = ST.last_chat
    texts = text_parts(req)
    check("метки картинок сквозные [Image 1..3]",
          texts.count("[Image 1]") == 1 and texts.count("[Image 2]") == 1 and texts.count("[Image 3]") == 1,
          str(texts))
    check("видео отдельным блоком [Video 1]", "[Video 1]" in texts, str(texts))
    check("кадров всего: 3 картинок + 4 видео = 7", len(media_parts(req)) == 7, str(len(media_parts(req))))
    check("порядок: промпт -> [Image 1..3] -> [Video 1]",
          texts[0] == "Опиши." and texts.index("[Image 1]") < texts.index("[Video 1]"))
    check("кадры - data:image/jpeg;base64",
          all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in media_parts(req)))
    # Лимит max_images НА КАЖДЫЙ пин: 2 пина по 8 кадров, лимит 3 -> 3+3 = 6
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж",
                     images={"img_0": make_fake_image_batch(8), "img_1": make_fake_image_batch(8)},
                     videos=None, media=make_media(max_images=3, max_video_frames=0))
    texts = text_parts(ST.last_chat)
    check("max_images=3 НА КАЖДЫЙ из 2 пинов -> 6 кадров (3+3)",
          len(media_parts(ST.last_chat)) == 6, str(len(media_parts(ST.last_chat))))
    check("метки сквозные [Image 1..6]", all(f"[Image {i}]" in texts for i in range(1, 7)), str(texts))
    check("в логе 'лимит max_images НА ПИН'", "НА ПИН" in log, log[-300:])
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж",
                     images=None, videos={"vid_0": make_fake_image_batch(2),
                                          "vid_1": make_fake_image_batch(2)},
                     media=make_media(max_video_frames=8))
    texts = text_parts(ST.last_chat)
    check("два видео-пина -> [Video 1]/[Video 2]",
          "[Video 1]" in texts and "[Video 2]" in texts, str(texts))
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж",
                     images=None, videos={"vid_0": make_fake_image_batch(12),
                                          "vid_1": make_fake_image_batch(12)},
                     media=make_media(max_video_frames=8))
    texts = text_parts(ST.last_chat)
    check("max_video_frames=8 НА КАЖДЫЙ видео-пин -> 8+8=16 кадров",
          "[Video 1]" in texts and "[Video 2]" in texts
          and len(media_parts(ST.last_chat)) == 16, str(texts))
    check("в логе 'лимит max_video_frames НА ПИН'", "НА ПИН" in log, log[-300:])

    # Метки времени видео ('<X.X seconds>' - формат llama.cpp при декоде видео, mtmd-helper)
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж",
                     images=None, videos={"vid_0": make_fake_image_batch(12)},
                     media=make_media(max_images=0, max_video_frames=6, video_fps=0))
    texts = text_parts(ST.last_chat)
    check("video_fps=0 - без меток времени", not any(t.endswith("seconds>") for t in texts),
          str(texts))
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж",
                     images=None, videos={"vid_0": make_fake_image_batch(12)},
                     media=make_media(max_images=0, max_video_frames=6, video_fps=2))
    texts = text_parts(ST.last_chat)
    marks = [t for t in texts if t.endswith("seconds>")]
    check("video_fps=2 - метки от ИСХОДНЫХ индексов (0,2,4,7,9,11)/2",
          marks == ["<0.0 seconds>", "<1.0 seconds>", "<2.0 seconds>", "<3.5 seconds>",
                    "<4.5 seconds>", "<5.5 seconds>"], str(marks))
    req_parts = ST.last_chat["messages"][-1]["content"]
    mark_pos = [i for i, p in enumerate(req_parts)
                if p.get("type") == "text" and p.get("text", "").endswith("seconds>")]
    img_pos = [i for i, p in enumerate(req_parts) if p.get("type") == "image_url"]
    check("каждая метка стоит прямо перед своим кадром",
          len(mark_pos) == 6 and all(m + 1 in img_pos for m in mark_pos), str(mark_pos))
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж",
                     images={"img_0": make_fake_image_batch(3)}, videos=None,
                     media=make_media(video_fps=2))
    texts = text_parts(ST.last_chat)
    check("video_fps не ставит метки на картинки", not any(t.endswith("seconds>") for t in texts),
          str(texts))
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж",
                     images={"img_0": make_fake_image_batch(3)}, videos=None)
    check("без медиа-выборки (0) все кадры уходят", len(media_parts(ST.last_chat)) == 3)
    check("медиа не попадают в историю ноды Memory", True)   # проверяется в [7]

    # ---------- 5b. префлайт: кадры подгоняются под контекст движка ----------
    print("\n[5b] Префлайт контекста: кадры режутся ДО отправки (двигок не обрезает)")
    nodes_mod._SERVER_CTX_CACHE.clear()
    ST.ctx_n = 700                            # кадр 48x72 ~ 18 ток. (с overhead); room = 700-1-128-512 = 59
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж",
                       videos={"vid_0": make_fake_image_batch(8)},
                       media=make_media(max_video_frames=8), max_tokens=0)
    check("контекст 700: 8 кадров -> урезано до 3", len(media_parts(ST.last_chat)) == 3,
          str(len(media_parts(ST.last_chat))))
    check("в логе заметка о подгонке 'кадров 8 -> 3'", "кадров 8 -> 3" in log, log)
    nodes_mod._SERVER_CTX_CACHE.clear()
    ST.ctx_n = 1400                           # текст 5000 симв ~ 1429 ток. -> room < 0
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж" * 5000,
                       videos={"vid_0": make_fake_image_batch(4)},
                       media=make_media(max_video_frames=4), max_tokens=0)
    check("текст занимает всё окно -> кадры убраны", len(media_parts(ST.last_chat)) == 0,
          str(len(media_parts(ST.last_chat))))
    check("в логе 'все кадры убраны'", "все кадры убраны" in log or "кадров 4 -> 0" in log, log)
    nodes_mod._SERVER_CTX_CACHE.clear()
    ST.ctx_n = 32768

    # ---------- 5c. 400 'exceeds the context window' -> человекочитаемый совет ----------
    print("\n[5c] Ошибка окна контекста от сервера -> понятное сообщение с советами")
    ST.ctx_overflow = True
    try:
        capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж")
        check("400 превышения контекста -> ошибка ноды", False)
    except nodes_mod.PolyStrataError as e:
        check("400 превышения контекста -> ошибка ноды",
              "НА КАЖДЫЙ пин" in str(e) and "max_frame_side" in str(e) and "118046" in str(e))
    ST.ctx_overflow = False

    # ---------- 5d. interrupt: Cancel ComfyUI прерывает генерацию ----------
    print("\n[5d] Interrupt (Cancel ComfyUI): исключение останавливает ноду, соединение закрывается")
    Interrupted = MM_STUB.InterruptProcessingException
    # (а) прерывание ДО запроса - сервер вообще не дёргается
    MM_STUB.interrupt_current_processing(True)
    ST.last_chat = None
    try:
        capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж")
        check("interrupt до запроса -> InterruptProcessingException", False)
    except Interrupted:
        check("interrupt до запроса -> InterruptProcessingException", True)
    check("interrupt до запроса: сервер не вызывался", ST.last_chat is None)
    # (б) прерывание ВО ВРЕМЯ «долгой генерации» - исключение приходит быстро, без ожидания ответа
    import time as _t
    ST.chat_delay = 30.0
    ST.disconnect_seen, ST.disconnect_dt = False, None
    buf = io.StringIO()
    result = {}

    def _run_long():
        try:
            nodes_mod.PolyStrataChat.execute(server=server, prompt="ДОЛГО")
        except BaseException as e:            # InterruptProcessingException - BaseException
            result["exc"] = e

    th = threading.Thread(target=_run_long, daemon=True)
    t0 = _t.monotonic()
    with contextlib.redirect_stdout(buf):
        th.start()
        _t.sleep(1.0)                         # запрос уже висит в поллинге stop_check
        MM_STUB.interrupt_current_processing(True)
        th.join(15)
    took = _t.monotonic() - t0
    check("interrupt во время генерации -> InterruptProcessingException быстро",
          isinstance(result.get("exc"), Interrupted) and took < 10.0,
          f"exc={result.get('exc')!r}, t={took:.1f}c (мок генерировал 30 c)")
    check("в логе 'ПРЕРВАНО (Cancel ComfyUI)'", "ПРЕРВАНО" in buf.getvalue(), buf.getvalue()[-300:])
    check("исключение НЕ поймано как обычная ошибка (BaseException дошёл до ноды)",
          type(result.get("exc")) is Interrupted)
    deadline = _t.monotonic() + 2.0          # сервер замечает EOF в своём цикле (шаг 0.05 c)
    while _t.monotonic() < deadline and not ST.disconnect_seen:
        _t.sleep(0.05)
    check("сервер УВИДЕЛ обрыв (EOF на сокете, как _watch_client) - и быстро: движок получает cancel",
          ST.disconnect_seen and ST.disconnect_dt is not None and ST.disconnect_dt < 2.0,
          f"seen={ST.disconnect_seen}, dt={ST.disconnect_dt}")
    ST.disconnect_seen, ST.disconnect_dt = False, None
    ST.chat_delay = 0.0
    _t.sleep(0.1)

    # ---------- 6. Sampling подключён к Chat: перекрытие + passthrough ----------
    print("\n[6] Sampling -> Chat: перекрытие и сквозная передача полей")
    smp = samp_node.run(min_p=0.1, repeat_penalty=1.15, repeat_last_n=128,
                        presence_penalty=0.2, frequency_penalty=-1.0, stop="СТОП",
                        logit_bias="42\n", extra_sampling='{"top_n_sigma": 1.0}')[0]
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ч",
                     temperature=0.8, top_p=0.95, reasoning="none", seed=9, sampling=smp)
    req = ST.last_chat
    check("min_p/repeat/repeat_last_n ушли (top_k в Sampling больше нет)",
          req.get("top_k") is None and req.get("min_p") == 0.1
          and req.get("repetition_penalty") == 1.15 and req.get("penalty_last_n") == 128)
    check("presence/frequency/stop ушли; seed - из виджета чата",
          req.get("presence_penalty") == 0.2 and req.get("frequency_penalty") is None
          and req.get("stop") == ["СТОП"] and req.get("seed") == 9)
    check("logit_bias ушёл в запрос", req.get("logit_bias") == {"42": -100.0})
    check("extra_sampling ушёл в запрос как есть", req.get("top_n_sigma") == 1.0)
    check("reasoning=none из чата -> оба механизма в запросе",
          req.get("reasoning_effort") == "none"
          and req.get("chat_template_kwargs") == {"enable_thinking": False})
    check("sampling НЕ перекрывает temperature (в Sampling её нет)",
          req.get("temperature") == 0.8)
    check("messages/stream не переопределяются extra", req.get("stream") is False)

    # ---------- 7. Memory ----------
    print("\n[7] Memory: история, очистка, лимит")
    out, _ = capture(mem_node.run, max_turns=2, system_prompt_text="СИСТЕМА")
    store = out[0]
    check("пустая история", store.history() == [] and store.turns() == 0)
    store.remember("в1", "о1")
    store.remember("в2", "о2")
    store.remember("в3", "о3")
    check("подрезка до max_turns=2", store.turns() == 2 and store.history()[0]["content"] == "в2")
    out, _ = capture(mem_node.run, max_turns=2, system_prompt_text="СИСТЕМА", clear=True)
    store = out[0]
    check("clear очищает", store.history() == [])
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="НОВЫЙ",
                     memory=store, images={"img_0": make_fake_image_batch(1)})
    req = ST.last_chat
    check("system от Memory перекрывает чат", req["messages"][0]["content"] == "СИСТЕМА")
    check("в историю пишется только текст (медиа не копятся)",
          all(isinstance(m["content"], str) for m in req["messages"][:-1]))
    check("ход записан в память", store.turns() == 1)

    # ---------- 7b. Пресеты системных промптов (как в llama.cpp-нодах) ----------
    print("\n[7b] Пресеты: models/LLM/prompts (.txt), перекрывание, обновление списка")
    proot = nodes_mod.prompt_root()
    check("папка пресетов = models/LLM/prompts",
          proot.name == "prompts" and proot.parent.name == "LLM", str(proot))
    opts = nodes_mod.system_prompt_options()
    check("список начинается с 'none'", opts[0] == nodes_mod.NO_SYSTEM_PROMPT)
    check("стандартные пресеты досозданы (как в llama.cpp-нодах)",
          "detailed_analysis.txt" in opts and "tags.txt" in opts, str(opts[:4]))
    # перекрывание: выбран пресет -> ТОЛЬКО он; none -> ручной текст
    manual = "РУЧНОЙ_СИСТЕМНЫЙ"
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж1",
                     system_prompt_preset="short_story.txt", system_prompt_text=manual)
    sys_msg = ST.last_chat["messages"][0]["content"]
    check("выбранный пресет ПЕРЕКРЫВАЕТ ручной текст",
          sys_msg == nodes_mod.DEFAULT_PROMPTS["short_story.txt"].strip(), sys_msg[:60])
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж2",
                     system_prompt_preset=nodes_mod.NO_SYSTEM_PROMPT,
                     system_prompt_text=manual)
    check("none -> берётся system_prompt_text",
          ST.last_chat["messages"][0]["content"] == manual)
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж3",
                     system_prompt_preset="нет_такого.txt", system_prompt_text=manual)
    check("файл пресета удалён/отсутствует -> пусто (не падает, текст игнорируется)",
          "system" not in [m["role"] for m in ST.last_chat["messages"]])
    # обновление списка: сканирование БЕЗ кеша при каждом вызове
    (proot / "subdir_test").mkdir(exist_ok=True)
    (proot / "subdir_test" / "fresh_test.txt").write_text("СВЕЖИЙ ПРЕСЕТ", encoding="utf-8")
    opts2 = nodes_mod.system_prompt_options()
    check("новый .txt (в т.ч. в подпапке) виден сразу - без кеша folder_paths",
          "subdir_test/fresh_test.txt" in opts2, str(opts2[-3:]))
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж4",
                     system_prompt_preset="subdir_test/fresh_test.txt")
    check("пресет из подпапки читается (относительный путь)",
          ST.last_chat["messages"][0]["content"] == "СВЕЖИЙ ПРЕСЕТ")
    mem2 = nodes_mod.PolyStrataMemory()
    out, _ = capture(mem2.run, max_turns=2, system_prompt_preset="subdir_test/fresh_test.txt")
    check("Memory тоже берёт пресет", out[0].system_line() == "СВЕЖИЙ ПРЕСЕТ")

    # ---------- 8. reserve VRAM -> EXTRA_RESERVED_VRAM (метод llama-нод) ----------
    print("\n[8] reserve VRAM: фактическое потребление -> EXTRA_RESERVED_VRAM")
    MM_STUB.EXTRA_RESERVED_VRAM = 600 * 1024 * 1024
    ST.sleeping = False
    ST.last_vram = None
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж",
                       reserve_mib=8000)
    check("/v1/vram вызван с reserve = 8000 (extra_mib больше нет)",
          ST.last_vram.get("reserve_mib") == 8000, str(ST.last_vram))
    # мок: freed = min(8000, 4000) = 4000, free = 4700, total = 32608 -> held = 27908
    held = TOTAL_GPU_MIB - 4700
    want = held * 1048576
    check("EXTRA_RESERVED_VRAM = (total - факт.свободно)",
          MM_STUB.EXTRA_RESERVED_VRAM == want,
          f"{MM_STUB.EXTRA_RESERVED_VRAM / 1048576:.0f} != {want / 1048576:.0f} MiB")
    check("в логе: 'PolyStrata держит ... MiB; ComfyUI: EXTRA_RESERVED_VRAM = ...'",
          "PolyStrata держит" in log and "EXTRA_RESERVED_VRAM" in log)
    check("в логе ЧАСТИЧНОЕ ОСВОБОЖДЕНИЕ (8000 > 4000 экспертов)", "ЧАСТИЧНОЕ" in log)
    check("запрошенная цель НЕ записана в EXTRA (16000-подобные цифры не попадают)",
          MM_STUB.EXTRA_RESERVED_VRAM != 8000 * 1048576)
    # снятие брони - нодой Unload
    out, _ = capture(unload_node.run, server=server, wait_s=1, any_input="СКВОЗНОЙ")
    check("Unload: сквозной выход", out["result"] == ("СКВОЗНОЙ",))
    check("Unload вернул EXTRA_RESERVED_VRAM ComfyUI",
          MM_STUB.EXTRA_RESERVED_VRAM == 600 * 1024 * 1024,
          f"{MM_STUB.EXTRA_RESERVED_VRAM / 1048576:.0f} MiB")
    # спящая модель: на GPU ничего нет - наша бронь снимается (сервер сам разбудится позже)
    ST.sleeping = True                      # сервер сам выгрузил модель (idle)
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж",
                       reserve_mib=4096)
    check("спящая модель: EXTRA_RESERVED_VRAM возвращён исходный",
          MM_STUB.EXTRA_RESERVED_VRAM == 600 * 1024 * 1024,
          f"{MM_STUB.EXTRA_RESERVED_VRAM / 1048576:.0f} MiB")
    check("спящая модель: резерв записан ('применится при загрузке')", "модель спит" in log)
    ST.sleeping = False
    # reserve=0: никаких запросов, бронь не трогается (инвариант 4)
    MM_STUB.EXTRA_RESERVED_VRAM = 900 * 1048576
    ST.last_vram = None
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж", reserve_mib=0)
    check("reserve=0: /v1/vram НЕ вызывался", ST.last_vram is None)
    check("reserve=0: бронь ComfyUI не тронута", MM_STUB.EXTRA_RESERVED_VRAM == 900 * 1048576)
    # ошибка резерва (400 --vram-elastic) не роняет генерацию и не меняет бронь
    ST.elastic = False
    MM_STUB.EXTRA_RESERVED_VRAM = 901 * 1048576
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж",
                       reserve_mib=2048)
    ST.elastic = True
    check("400 --vram-elastic не роняет генерацию", out[1][0] == "МОК-ОТВЕТ: всё вижу.", str(out[1]))
    check("ошибка резерва не трогает бронь", MM_STUB.EXTRA_RESERVED_VRAM == 901 * 1048576)
    check("в логе РЕЗЕРВ НЕ ПРИМЕНЁН", "РЕЗЕРВ НЕ ПРИМЕНЁН" in log)

    # ---------- 8b. Авто-выгрузка: сервер-сирота (байпасс/другой воркфлоу) ----------
    print("\n[8b] Авто-выгрузка: главная нода забайпашена / другой воркфлоу (механика llama-нод)")
    nodes_mod.AUTO_UNLOAD_WAIT_S = 1.0           # в тестах не ждём 30 c
    nodes_mod.ACTIVE_SERVERS.clear()
    MM_STUB.EXTRA_RESERVED_VRAM = 600 * 1024 * 1024               # исходник (перед 8b мог быть 901)
    nodes_mod.update_global_vram_reservation(555 * 1024 * 1024)   # бронь, выставленная нодой Chat
    ST.sleeping = False
    base = nodes_mod.PolyStrataClient(server, "").base
    # 1) прогон чата регистрирует сервер
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж")
    check("чат регистрирует сервер для авто-выгрузки", base in nodes_mod.ACTIVE_SERVERS,
          str(nodes_mod.ACTIVE_SERVERS))
    # 2) воркфлоу с ЖИВОЙ нодой Chat на этом сервере - не выгружаем
    prompt = {"1": {"class_type": "PolyStrataChat", "inputs": {"server": server}}}
    nodes_mod._auto_unload_for_prompt(prompt)
    check("живая нода Chat в воркфлоу -> модель не выгружается", ST.sleeping is False)
    check("сервер остаётся под наблюдением", base in nodes_mod.ACTIVE_SERVERS)
    # 3) байпасс (новый фронт: BypassedNode__) -> выгрузка
    prompt_bypass = {"1": {"class_type": "BypassedNode__PolyStrataChat",
                           "inputs": {"server": server}}}
    nodes_mod._auto_unload_for_prompt(prompt_bypass)
    check("главная нода забайпашена -> POST /v1/unload (модель спит)", ST.sleeping is True)
    check("сервер снят с наблюдения", base not in nodes_mod.ACTIVE_SERVERS)
    check("EXTRA_RESERVED_VRAM возвращён ComfyUI при авто-выгрузке",
          MM_STUB.EXTRA_RESERVED_VRAM == 600 * 1024 * 1024,
          f"{MM_STUB.EXTRA_RESERVED_VRAM / 1048576:.0f} MiB")
    # 4) другой воркфлоу (наших нод нет вообще) -> тоже выгрузка
    ST.sleeping = False
    nodes_mod.ACTIVE_SERVERS.add(base)
    nodes_mod.update_global_vram_reservation(555 * 1024 * 1024)
    nodes_mod._auto_unload_for_prompt({"9": {"class_type": "KSampler", "inputs": {}}})
    check("другой воркфлоу без наших нод -> выгрузка", ST.sleeping is True)
    check("EXTRA снова исходный", MM_STUB.EXTRA_RESERVED_VRAM == 600 * 1024 * 1024)
    # 5) старый фронт: байпасс как mode=4 -> нода не живая
    ST.sleeping = False
    nodes_mod.ACTIVE_SERVERS.add(base)
    nodes_mod._auto_unload_for_prompt(
        {"1": {"class_type": "PolyStrataChat", "mode": 4, "inputs": {"server": server}}})
    check("mode=4 (старый фронт, байпасс) -> выгрузка", ST.sleeping is True)
    # 6) нода Unload в воркфлоу тоже держит сервер (не выгружаем из-под неё)
    prompt_un = {"1": {"class_type": "PolyStrataUnload", "inputs": {"server": server}}}
    nodes_mod.ACTIVE_SERVERS.add(base)
    ST.sleeping = False
    nodes_mod._auto_unload_for_prompt(prompt_un)
    check("живая нода Unload -> модель не выгружается из-под неё", ST.sleeping is False)
    nodes_mod.ACTIVE_SERVERS.discard(base)
    # 7) сервер выгрузить не удалось (недоступен) - остаётся под наблюдением
    nodes_mod.ACTIVE_SERVERS.add("127.0.0.1:1")
    nodes_mod._auto_unload_for_prompt({"1": {"class_type": "KSampler", "inputs": {}}})
    check("недоступный сервер: ошибка в лог, остаётся под наблюдением",
          "127.0.0.1:1" in nodes_mod.ACTIVE_SERVERS)
    nodes_mod.ACTIVE_SERVERS.discard("127.0.0.1:1")
    # 8) хук unload_all_models: выгрузить ВСЁ наблюдаемое
    ST.sleeping = False
    nodes_mod.ACTIVE_SERVERS.add(base)
    nodes_mod._auto_unload_all()
    check("unload_all_models: сервер выгружен и снят с наблюдения",
          ST.sleeping is True and base not in nodes_mod.ACTIVE_SERVERS)
    # 9) наблюдение пусто - тихий no-op
    nodes_mod.ACTIVE_SERVERS.clear()
    check("нет наблюдаемых серверов -> ничего не делает",
          nodes_mod._auto_unload_for_prompt({}) == [])
    ST.sleeping = False

    # ---------- 8c. Очистка памяти ComfyUI: тоггл clear_comfy_vram + листы на входах ----------
    print("\n[8c] Очистка памяти ComfyUI (тоггл): один раз на запуск воркфлоу, листы на входах")
    schema2 = nodes_mod.PolyStrataChat.define_schema()
    ins2 = {i.name: i for i in schema2["inputs"]}
    check("V3: виджет-тоггл clear_comfy_vram есть (BOOLEAN)",
          ins2.get("clear_comfy_vram") is not None and ins2["clear_comfy_vram"].kind == "BOOLEAN",
          str(sorted(ins2)))
    check("V3: clear_comfy_vram default=False (выкл - не трогаем ComfyUI)",
          ins2["clear_comfy_vram"].kw.get("default") is False)
    hooks_was = nodes_mod._AUTO_HOOKS["installed"]
    try:
        nodes_mod._AUTO_HOOKS["installed"] = True       # как в реальном ComfyUI (хук execute сбрасывает флаг)
        MM_STUB.unload_calls["n"] = 0
        MM_STUB.cache_calls["n"] = 0
        nodes_mod.note_workflow_start()                 # начало «запуска воркфлоу»
        out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="чисти")
        check("тоггл выключен: unload_all_models НЕ вызывается",
              MM_STUB.unload_calls["n"] == 0 and MM_STUB.cache_calls["n"] == 0,
              str((MM_STUB.unload_calls, MM_STUB.cache_calls)))
        out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="чисти",
                           clear_comfy_vram=True)
        check("тоггл включён: unload_all_models + torch-кэш, генерация работает",
              MM_STUB.unload_calls["n"] == 1 and MM_STUB.cache_calls["n"] == 1
              and out[1][0] == "МОК-ОТВЕТ: всё вижу.", str((MM_STUB.unload_calls, MM_STUB.cache_calls)))
        check("в логе 'очистка памяти ComfyUI перед генерацией'",
              "очистка памяти ComfyUI перед генерацией" in log, log[-400:])
        n1 = MM_STUB.unload_calls["n"]
        out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="чисти ещё",
                           clear_comfy_vram=True)
        check("второе исполнение в ТОМ ЖЕ запуске (лист/цикл/сид) - очистка НЕ повторяется",
              MM_STUB.unload_calls["n"] == n1, f"{MM_STUB.unload_calls['n']} != {n1}")
        check("в логе 'уже сделана в этом запуске воркфлоу - пропуск'",
              "уже сделана в этом запуске воркфлоу" in log, log[-400:])
        nodes_mod.note_workflow_start()                 # хук execute: новый запуск воркфлоу
        out, log = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="новый запуск",
                           clear_comfy_vram=True)
        check("новый запуск воркфлоу -> очистка снова доступна",
              MM_STUB.unload_calls["n"] == n1 + 1, str(MM_STUB.unload_calls))
        # Полифорк: цикл = список картин в ОДНОМ запросе - очистка и тут одна
        n2 = MM_STUB.unload_calls["n"]
        out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="много картинок",
                         clear_comfy_vram=True,
                         images={"img_0": make_fake_image_batch(4), "img_1": make_fake_image_batch(3)},
                         media=make_media(max_images=0))
        check("батч 4+3 картин за один запрос = одна очистка",
              len(media_parts(ST.last_chat)) == 7 and MM_STUB.unload_calls["n"] == n2,
              f"frames={len(media_parts(ST.last_chat))}, unloads={MM_STUB.unload_calls['n'] - n2}")
    finally:
        nodes_mod._AUTO_HOOKS["installed"] = hooks_was  # в тестах хуков нет (comfy недоступен)
        nodes_mod._COMFY_CLEAR["done"] = False
    # Запасной путь: хуков нет (comfy недоступен) - сбрасывать флаг некому, очистка на КАЖДОМ
    # вызове (никогда не 'залипает' в состоянии 'уже очищено')
    MM_STUB.unload_calls["n"] = 0
    capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж", clear_comfy_vram=True)
    capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="ж", clear_comfy_vram=True)
    check("без хуков (comfy недоступен) очистка не зависает: на каждом вызове",
          MM_STUB.unload_calls["n"] == 2, str(MM_STUB.unload_calls))

    # --- листы на входах (list-паки): скаляры - первый элемент, нода не падает
    out, log = capture(nodes_mod.PolyStrataChat.execute, server=server,
                       prompt=["первый промпт", "второй промпт"], seed=[123, 456],
                       max_tokens=[256])
    req = ST.last_chat
    check("prompt-список -> первый элемент в запросе",
          req["messages"][-1]["content"] == "первый промпт", str(req["messages"][-1])[:200])
    check("seed-список -> первый seed", req.get("seed") == 123, str(req.get("seed")))
    check("max_tokens-список -> первый", req.get("max_tokens") == 256)
    check("в логе 'взят первый элемент'", "взят первый элемент" in log, log[-400:])
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=[server, "127.0.0.1:1"],
                     prompt="жс2")
    check("server-список -> первый адрес, запрос прошёл",
          out[1][0] == "МОК-ОТВЕТ: всё вижу.", str(out[1]))
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="жс3",
                     images=[make_fake_image_batch(2), make_fake_image_batch(3)],
                     media=make_media(max_images=0))
    texts = text_parts(ST.last_chat)
    check("images-СПИСОК батчей -> все кадры одним запросом (2+3=5)",
          len(media_parts(ST.last_chat)) == 5, str(len(media_parts(ST.last_chat))))
    check("метки сквозные [Image 1..5]",
          all(f"[Image {i}]" in texts for i in range(1, 6)), str(texts))
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="жс4",
                     memory=[store], sampling=[smp])
    req = ST.last_chat
    check("memory-список -> история используется (первый элемент)",
          req["messages"][0]["content"] == "СИСТЕМА" and store.turns() == 2,
          f"{req['messages'][0]['content']!r}, turns={store.turns()}")
    check("sampling-список -> параметры уходят (min_p из суб-ноды)",
          req.get("min_p") == 0.1, str(req.get("min_p")))
    # --- универсальная нормализация (_scalars): списки СРАЗУ во ВСЕХ скалярных виджетах.
    # Имена нигде не перечисляются - весь kwargs прогоняется через нормализатор, так что
    # покрытие полное и распространяется на любые будущие виджеты без правок кода
    out, log = capture(nodes_mod.PolyStrataChat.execute,
                       server=[server], prompt=["списки везде"],
                       reasoning=["none"], max_tokens=[64], temperature=[0.5], top_p=[0.9],
                       top_k=[33], seed=[7], reserve_mib=[0],
                       system_prompt_preset=["none"], system_prompt_text=[""],
                       clear_comfy_vram=[False], media=[make_media(max_frame_side=512)])
    req = ST.last_chat
    check("списки во ВСЕХ скалярных виджетах сразу -> первые элементы, запрос прошёл",
          req["messages"][-1]["content"] == "списки везде" and req.get("seed") == 7
          and req.get("max_tokens") == 64 and req.get("temperature") == 0.5
          and req.get("reasoning_effort") == "none" and req.get("top_k") == 33
          and out[1][0] == "МОК-ОТВЕТ: всё вижу.",
          str({k: req.get(k) for k in ("seed", "max_tokens", "temperature", "reasoning_effort",
                                       "top_k")}))

    # ---------- 9. Unload ----------
    print("\n[9] Unload: 409 без wait, выгрузка, сброс серверного резерва")
    ST.sleeping = False
    try:
        capture(unload_node.run, server=server, wait_s=0)
        check("409 model_busy без wait -> ошибка", False)
    except nodes_mod.PolyStrataError as e:
        check("409 model_busy без wait -> ошибка", "409" in str(e))
    out, log = capture(unload_node.run, server=server, wait_s=1, any_input=123)
    check("выгрузка прошла, сквозной any_input", out["result"] == (123,))
    check("сервер после unload спит", ST.sleeping is True)
    check("серверный резерв сброшен (reserve_mib=null)", "сброшен" in out["ui"]["text"][0])
    check("EXTRA_RESERVED_VRAM возвращён (в ui-тексте)", "EXTRA_RESERVED_VRAM" in out["ui"]["text"][0])
    # пробуждение следующим чат-запросом
    ST.sleeping = False
    out, _ = capture(nodes_mod.PolyStrataChat.execute, server=server, prompt="разбуди")
    check("следующий чат работает (модель проснулась сама)", out[1][0] == "МОК-ОТВЕТ: всё вижу.", str(out[1]))

    # ---------- 10. V1-фолбэк (ComfyUI без comfy_api) ----------
    print("\n[10] V1-фолбэк: Chat без schema V3")
    for stale in ("ComfyUI-PolyStrata", "ComfyUI-PolyStrata.nodes", "ComfyUI-PolyStrata.ps_client"):
        sys.modules.pop(stale, None)         # свежая загрузка без кэша
    saved = block_comfy_api()
    try:
        pack_v1 = load_pack(as_package=True)
        nv1 = _il.import_module("ComfyUI-PolyStrata.nodes")
        check("_HAS_V3=False без comfy_api", nv1._HAS_V3 is False)
        check("Chat - V1-класс (INPUT_TYPES)", hasattr(nv1.PolyStrataChat, "INPUT_TYPES"))
        cin = nv1.PolyStrataChat.INPUT_TYPES()
        check("V1: входы images/videos (IMAGE) отдельно",
              cin["optional"]["images"][0] == "IMAGE" and cin["optional"]["videos"][0] == "IMAGE")
        check("V1: seed/top_k/model в чате, api_key/extra_mib/медиа-лимитов НЕТ",
              "seed" in cin["required"] and "top_k" in cin["required"]
              and "model" in cin["required"]
              and not ({"api_key", "extra_mib", "max_frame_side", "max_images",
                        "max_video_frames", "video_fps"} & set(cin["required"])))
        check("V1: model - сентинел первым в опциях (default), виджет после server",
              cin["required"]["model"][0][0] == nv1.MODEL_SWITCH_OFF
              and cin["required"]["model"][1].get("default") == nv1.MODEL_SWITCH_OFF
              and list(cin["required"]).index("server") < list(cin["required"]).index("model"))
        check("V1: media - опциональный вход PS_MEDIA",
              cin["optional"]["media"][0] == nodes_mod.PS_MEDIA)
        check("V1: пресеты тоже есть (system_prompt_preset/system_prompt_text)",
              "system_prompt_preset" in cin["required"]
              and "system_prompt_text" in cin["required"])
        check("V1: тоггл clear_comfy_vram тоже есть (BOOLEAN, default False)",
              "clear_comfy_vram" in cin["required"]
              and cin["required"]["clear_comfy_vram"][0] == "BOOLEAN"
              and cin["required"]["clear_comfy_vram"][1].get("default") is False)
        check("V1: выходы text+thoughts", nv1.PolyStrataChat.RETURN_TYPES == ("STRING", "STRING"))
        v1 = nv1.PolyStrataChat()
        ST.load_calls.clear()
        out, _ = capture(v1.generate, server=server, prompt="V1-смена",
                         model="sage-a1.5-ablitrex-abliterated-bf16.i1-q4_k_m")
        check("V1: переключение тоже работает (POST /api/models/load)",
              ST.load_calls and ST.load_calls[-1] == {"path": "/api/models/load",
                                                      "model": "sage-a1.5-ablitrex-abliterated-bf16.i1-q4_k_m"}
              and ST.current == "sage-a1.5-ablitrex-abliterated-bf16.i1-q4_k_m"
              and "model" not in ST.last_chat, str(ST.load_calls))
        ST.current = "gemma-4-26b-a4b"
        out, _ = capture(v1.generate, server=server, prompt="V1-тест", seed=77,
                         videos=make_fake_image_batch(2), media=make_media(max_video_frames=2))
        check("V1 generate работает", out == ("МОК-ОТВЕТ: всё вижу.", ST.thoughts), str(out))
        check("V1: seed из виджета чата ушёл", ST.last_chat.get("seed") == 77)
        texts = text_parts(ST.last_chat)
        check("V1: видео тоже блоком [Video 1]", "[Video 1]" in texts, str(texts))
        out, _ = capture(v1.generate, server=server, prompt="V1-fps",
                         videos=make_fake_image_batch(12),
                         media=make_media(max_video_frames=6, video_fps=2))
        texts = text_parts(ST.last_chat)
        check("V1: video_fps -> метки '<X.X seconds>' в запросе",
              [t for t in texts if t.endswith("seconds>")] ==
              ["<0.0 seconds>", "<1.0 seconds>", "<2.0 seconds>", "<3.5 seconds>",
               "<4.5 seconds>", "<5.5 seconds>"], str(texts))
        before = MM_STUB.EXTRA_RESERVED_VRAM
        ST.free_mib = 700                     # модель "загружена" и держит память
        out, _ = capture(v1.generate, server=server, prompt="V1-reserve", reserve_mib=1024)
        check("V1: бронь EXTRA работает и здесь",
              MM_STUB.EXTRA_RESERVED_VRAM == (TOTAL_GPU_MIB - ST.free_mib) * 1048576,
              f"{MM_STUB.EXTRA_RESERVED_VRAM / 1048576:.0f} MiB")
        nv1.update_global_vram_reservation(0)
        check("V1: снятие брони", MM_STUB.EXTRA_RESERVED_VRAM == before,
              f"{MM_STUB.EXTRA_RESERVED_VRAM / 1048576:.0f} != {before / 1048576:.0f}")
        # V1: тоггл очистки памяти ComfyUI работает так же (хуков в тестах нет - путь на каждый вызов)
        MM_STUB.unload_calls["n"] = 0
        MM_STUB.cache_calls["n"] = 0
        out, log = capture(v1.generate, server=server, prompt="V1-очистка", clear_comfy_vram=True)
        check("V1: тоггл clear_comfy_vram выгружает модели ComfyUI",
              MM_STUB.unload_calls["n"] == 1 and MM_STUB.cache_calls["n"] == 1
              and out[0] == "МОК-ОТВЕТ: всё вижу.", str((MM_STUB.unload_calls, MM_STUB.cache_calls)))
        check("V1: в логе 'очистка памяти ComfyUI'", "очистка памяти ComfyUI" in log, log[-300:])
        # V1: листы на входах тоже обрабатываются
        out, _ = capture(v1.generate, server=server, prompt=["V1-лист", "второй"],
                         seed=[55, 66], images=[make_fake_image_batch(2), make_fake_image_batch(1)],
                         media=make_media(max_images=0))
        check("V1: prompt/seed-листы -> первые элементы",
              ST.last_chat.get("seed") == 55
              and text_parts(ST.last_chat)[0] == "V1-лист",
              f"seed={ST.last_chat.get('seed')}, texts={text_parts(ST.last_chat)[:2]}")
        check("V1: images-список батчей -> 3 кадра одним запросом",
              len(media_parts(ST.last_chat)) == 3, str(len(media_parts(ST.last_chat))))
    finally:
        restore_comfy_api(saved)
        for stale in ("ComfyUI-PolyStrata", "ComfyUI-PolyStrata.nodes", "ComfyUI-PolyStrata.ps_client"):
            sys.modules.pop(stale, None)

    # ---------- 11. клиент ----------
    print("\n[11] Поли-клиент: адрес, ошибки, фильтры")
    ps = nodes_mod.PolyStrataClient
    check("адрес без порта -> 8080", ps("127.0.0.1").base == "http://127.0.0.1:8080")
    check("http:// и слэш нормализуются", ps(" http://1.2.3.4:9000/ ").base == "http://1.2.3.4:9000")
    check("пустой адрес -> ошибка",
          bool(_try(lambda: ps("  ").base)))
    cli = ps(server, "")
    out, _ = capture(cli.models)
    check("клиент: GET /api/models -> current + записи реестра",
          out.get("current") == ST.current and isinstance(out.get("models"), list)
          and len(out["models"]) == 4 and "id" in out["models"][0], str(out)[:160])
    out, _ = capture(cli.switch_model, "sage-a1.5-ablitrex-abliterated-bf16.i1-q4_k_m")
    check("клиент: switch_model -> {'status': 'loaded', 'model': {запись}}",
          out.get("status") == "loaded"
          and out.get("model", {}).get("id") == "sage-a1.5-ablitrex-abliterated-bf16.i1-q4_k_m"
          and ST.current == "sage-a1.5-ablitrex-abliterated-bf16.i1-q4_k_m", str(out)[:160])
    ST.current = "gemma-4-26b-a4b"
    out, _ = capture(cli.chat, [{"role": "user", "content": "п"}], stop=["а", "б", "в", "г", "д"])
    check("stop > 4 строк -> 4", len(ST.last_chat.get("stop", [])) == 4)
    out, _ = capture(cli.chat, [{"role": "user", "content": "п"}], top_k=0, seed=0)
    check("top_k=0/seed=0 не отправляются", "top_k" not in ST.last_chat and "seed" not in ST.last_chat)
    out, _ = capture(cli.chat, [{"role": "user", "content": "п"}], logit_bias={"5": -100.0},
                     extra={"messages": "ВЗЛОМ", "stream": True, "custom": 1})
    check("logit_bias уходит", ST.last_chat.get("logit_bias") == {"5": -100.0})
    check("extra не переопределяет messages/stream, остальное проходит",
          isinstance(ST.last_chat.get("messages"), list) and ST.last_chat.get("stream") is False
          and ST.last_chat.get("custom") == 1)
    capture(cli.chat, [{"role": "user", "content": "п"}], reasoning="none")
    check("reasoning=none -> оба механизма (клиент)",
          ST.last_chat.get("reasoning_effort") == "none"
          and ST.last_chat.get("chat_template_kwargs") == {"enable_thinking": False})
    # контекст: status -> context.native; если его нет - запасной путь /props
    nodes_mod._SERVER_CTX_CACHE.clear()
    ST.ctx_n, ST.props_ctx = 0, 16384
    check("контекст недоступен в status -> берётся из /props",
          nodes_mod._server_ctx(ps(server, "")) == 16384)
    ST.ctx_n, ST.props_ctx = 32768, 0

    # ---------- итог ----------
    print(f"\n{'=' * 60}")
    if FAILS:
        print(f"ПРОВАЛЕНО {len(FAILS)}: {FAILS}")
        sys.exit(1)
    print("ВСЕ ПРОВЕРКИ ЗЕЛЁНЫЕ")


def _try(fn):
    try:
        fn()
        return False
    except Exception:
        return True


if __name__ == "__main__":
    main()
