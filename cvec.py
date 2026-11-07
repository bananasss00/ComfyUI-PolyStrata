"""cvec.py - движковая часть ноды «PolyStrata Control Vector» (ComfyUI-PolyStrata).

Что такое control vector (проверено по исходникам PolyStrata: serve/server.py,
serve/models.py, serve/runconfig.py, setup.py и docs/DETAILS.md, раздел
«Experimental speed projection»; постановка задачи - флаги из реального конфига):

  Маленький GGUF (~0.5 МБ) с направлениями по слоям (тензоры ``direction.<N>``),
  обученный отдельно от модели. Флаги запуска ДВИЖКА пишутся в конфиг
  ``strata-<модель>.json`` (setup.py пишет ровно такую строку при выборе
  «experimental speed projection»):

      --control-vector-scaled <файл>.gguf:<скейл>   # можно повторять: эффекты складываются
      --control-vector-layer-range <start> <end>    # применять только к слоям из диапазона
      --cvec-mode project|add                       # project: h -= s*(h.v)*v («срезает» проекцию
                                                    #  на направление; s=1.0 - срезать целиком);
                                                    # add: h += s*v (аддитивно, малые s ~0.05-0.15)
      --cvec-dir per-layer|single:<L>               # свой вектор на каждый слой / один слой L на все

  Это СТАРТОВЫЕ параметры движка: в рантайме меняется только вкл/выкл уже
  загруженного вектора - поле ``"experimental_speed_projection": false`` в теле
  POST /v1/chat/completions (сервер: StrataEngine.projection_key, «Absent = on»).

Механика применения из ноды (важнейшее, проверено по serve/server.py):

  * POST /api/models/load перечитывает конфиг С ДИСКА только при переключении на
    ДРУГУЮ запись реестра (Service.switch_model -> _switch_to -> run_config);
  * для УЖЕ текущей модели load() - no-op (``if self.loaded(): return``), а
    unload+load рестартит движок со СТАРОЙ командной строкой (engine.restart()
    хранит её с момента создания движка).

  Поэтому нода НЕ правит базовый конфиг (файлы setup'а остаются нетронутыми),
  а пишет РЯДОМ вариант ``strata-<id>-<suffix>.json`` - клон базового конфига
  с cvec-флагами и своим ``model_name``. Реестр (serve/models.py discover,
    глоб strata-*.json, маска только с дефисом) подхватывает его как ещё одну
  модель, и переключение база <-> вариант идёт штатным hot-swap'ом сервера
  (POST /api/models/load). Полный откат - удалить файл варианта.

Здесь только чистая логика без HTTP (клиент - ps_client.PolyStrataClient):
GGUF-заголовок, сборка/вычистка флагов, скан папок на векторы, запись варианта.
"""
from __future__ import annotations

import json
import os
import re
import struct
import time
from pathlib import Path

# ------------------------------------------------------------------ константы флагов
# арность = сколько ЗНАЧЕНИЙ следует за флагом в списке args конфига.
CVEC_FLAG_ARITY = {
    "--control-vector": 1,               # стоковый llama.cpp (без скейла)
    "--control-vector-scaled": 1,        # "<файл>:<скейл>" одним токеном
    "--control-vector-layer-range": 2,  # "<start> <end>"
    "--cvec-mode": 1,                    # project | add (патч Strata/пакета вектора)
    "--cvec-dir": 1,                     # per-layer | single:<L> (патч Strata/пакета)
}

# маркер в варианте-конфиге: «этот файл создала нода ComfyUI-PolyStrata».
# Сервер и реестр неизвестных ключей не читают (проверено: describe() берёт свои
# ключи, run_config/load - json.loads + проверка dict), ключ безвреден.
CVEC_MARKER = "cvec_variant"

VECTOR_TENSOR = re.compile(r"^direction\.(\d+)$")

# ------------------------------------------------------------------ GGUF-заголовок
_GGUF_MAGIC = b"GGUF"
# gguf-py GGUFValueType: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool,
# 8 string, 9 array, 10 u64, 11 i64, 12 f64.
_SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_gguf_header(path, want_tensors: bool = True, interesting: tuple = ()):
    """Заголовок GGUF без чтения данных: (метаданные, имена тензоров).

    want_tensors=False - прочитать только метаданные и НЕ ходить по тензорам
    (для больших файлов моделей нужен только <arch>.block_count).
    interesting - ключи, значения которых возвращать (остальные пропускаются,
    но по файлу позиция всё равно проходит - формат последовательный).
    ValueError - не GGUF / битый заголовок / несуразные счётчики.
    """
    p = Path(path)
    with open(p, "rb") as f:
        def exact(n: int) -> bytes:
            b = f.read(n)
            if len(b) != n:
                raise ValueError(f"обрыв заголовка GGUF ({p.name})")
            return b

        def r_str() -> str:
            n = struct.unpack("<Q", exact(8))[0]
            if n > 1 << 22:
                raise ValueError(f"несуразная длина строки в GGUF ({p.name})")
            return exact(n).decode("utf-8", "replace")

        def skip_value(t: int):
            if t == 8:
                r_str()
            elif t == 9:                          # массив: тип элемента + счётчик
                et = struct.unpack("<I", exact(4))[0]
                n = struct.unpack("<Q", exact(8))[0]
                if n > 1 << 24:
                    raise ValueError(f"несуразный массив в GGUF ({p.name})")
                for _ in range(n):
                    skip_value(et)
            elif t in _SCALAR_SIZE:
                exact(_SCALAR_SIZE[t])
            else:
                raise ValueError(f"неизвестный тип значения GGUF {t} ({p.name})")

        def read_value(t: int):
            if t == 8:
                return r_str()
            if t == 7:
                return exact(1) != b"\x00"
            if t in (4, 10):
                return struct.unpack("<Q" if t == 10 else "<I", exact(_SCALAR_SIZE[t]))[0]
            if t in (5, 11):
                return struct.unpack("<q" if t == 11 else "<i", exact(_SCALAR_SIZE[t]))[0]
            if t == 6:
                return struct.unpack("<f", exact(4))[0]
            if t == 12:
                return struct.unpack("<d", exact(8))[0]
            if t in (0, 1, 2, 3):
                return int.from_bytes(exact(_SCALAR_SIZE[t]), "little", signed=(t % 2 == 1))
            if t == 9:
                et = struct.unpack("<I", exact(4))[0]
                n = struct.unpack("<Q", exact(8))[0]
                return [read_value(et) for _ in range(min(n, 4096))]  # для нужд ноды хватит начала
            raise ValueError(f"неизвестный тип значения GGUF {t}")

        if exact(4) != _GGUF_MAGIC:
            raise ValueError(f"не GGUF-файл (магия не совпала): {p.name}")
        version = struct.unpack("<I", exact(4))[0]
        if not 2 <= version <= 5:
            raise ValueError(f"неподдерживаемая версия GGUF {version}: {p.name}")
        n_tensors = struct.unpack("<Q", exact(8))[0]
        n_kv = struct.unpack("<Q", exact(8))[0]
        if n_kv > 1 << 16 or n_tensors > 1 << 20:
            raise ValueError(f"несуразные счётчики GGUF ({p.name})")

        want = set(interesting)
        meta: dict = {}
        for _ in range(n_kv):
            key = r_str()
            t = struct.unpack("<I", exact(4))[0]
            # "*.block_count" собираем всегда: ключ зависит от архитектуры, которая
            # встречается в KV раньше/позже него - так model_layer_count читает
            # заголовок ОДНИМ проходом (большой файл модели не перечитывать дважды)
            if key in want or key.endswith(".block_count"):
                meta[key] = read_value(t)
            else:
                skip_value(t)
        if not want_tensors:
            return meta, []

        names = []
        for _ in range(n_tensors):
            name = r_str()
            nd = struct.unpack("<I", exact(4))[0]
            if nd > 8:
                raise ValueError(f"несуразная размерность тензора в {p.name}")
            exact(nd * 8)          # dims
            exact(4 + 8)           # type + offset
            names.append(name)
        return meta, names


def control_vector_layers(path):
    """Слои, для которых в GGUF есть тензор direction.<N>: отсортированный список
    индексов; [] - GGUF читается, но direction-тензоров нет; None - не разобрать."""
    try:
        _, names = read_gguf_header(path, want_tensors=True)
    except (OSError, ValueError, struct.error):
        return None
    out = []
    for n in names:
        m = VECTOR_TENSOR.match(n)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


_LAYER_CACHE: dict = {}          # (путь, размер, mtime) -> block_count (модели)


def model_layer_count(path):
    """Число слоёв модели из GGUF-метаданных (<arch>.block_count); None - не узнать.
    Кеш по (путь, размер, mtime): нода OUTPUT_NODE - заголовок большого файла не
    перечитывать на каждый запуск воркфлоу."""
    p = Path(path)
    try:
        st = p.stat()
        key = (str(p), st.st_size, st.st_mtime_ns)
    except OSError:
        return None
    if key in _LAYER_CACHE:
        return _LAYER_CACHE[key]
    try:
        meta, _ = read_gguf_header(p, want_tensors=False,
                                   interesting=("general.architecture",))
        arch = str(meta.get("general.architecture") or "")
        bc = meta.get(arch + ".block_count") if arch else None
        out = int(bc) if isinstance(bc, (int, float)) and not isinstance(bc, bool) else None
    except (OSError, ValueError, struct.error):
        out = None
    _LAYER_CACHE[key] = out
    return out


# ------------------------------------------------------------------ флаги движка
def strip_cvec_args(args):
    """Убрать из args конфига ВСЕ control-vector-флаги вместе с их значениями
    (арность известна; значение, похожее на флаг «--», не съедаем - защита от
    битого конфига). Возвращает новый список."""
    out, i = [], 0
    src = [str(a) for a in (args or [])]
    while i < len(src):
        a = src[i]
        if a in CVEC_FLAG_ARITY:
            i += 1
            for _ in range(CVEC_FLAG_ARITY[a]):
                if i < len(src) and not src[i].startswith("--"):
                    i += 1
            continue
        out.append(a)
        i += 1
    return out


def fmt_scale(scale: float) -> str:
    """1.0 -> '1.0', 0.5 -> '0.5', -0.8 -> '-0.8' (как пишет setup: «...gguf:1.0»)."""
    s = f"{float(scale):.6g}"
    if "." not in s and "e" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s


def build_cvec_args(vector_path, scale, mode, dir_mode, single_layer,
                    layer_mode, layer_start, layer_end, extra_vectors,
                    strata_engine: bool):
    """Список args движка в каноническом порядке setup.py:
    scaled (+ доп. векторы) -> layer-range (только «вручную») -> cvec-mode -> cvec-dir.
    strata_engine=False (стоковый llama.cpp): только стоковые флаги scaled/layer-range
    (cvec-mode/cvec-dir - патчи пакета вектора, сток их не знает и не стартует)."""
    args = ["--control-vector-scaled", f"{vector_path}:{fmt_scale(scale)}"]
    for p, s in (extra_vectors or []):
        args += ["--control-vector-scaled", f"{p}:{fmt_scale(s)}"]
    if layer_mode == "manual":
        args += ["--control-vector-layer-range", str(int(layer_start)), str(int(layer_end))]
    if strata_engine:
        args += ["--cvec-mode", str(mode)]
        if str(dir_mode) == "per-layer":
            args += ["--cvec-dir", "per-layer"]
        else:
            args += ["--cvec-dir", f"single:{int(single_layer)}"]
    return args


def parse_live_cvec(value):
    """engine.info['cvec'] с GET /metrics (формат из serve/web/app.js:
    'project:4-44[:singleL]' | 'add:4-44' | 0) -> dict или None (вектора нет).
    Слои/одиночный слой - None, когда в строке их нет."""
    s = str(value if value is not None else "0")
    if s in ("0", "", "None"):
        return None
    parts = s.split(":")
    out = {"mode": parts[0], "start": None, "end": None, "single": None}
    if len(parts) >= 2 and "-" in parts[1]:
        a, b = parts[1].split("-", 1)
        try:
            out["start"], out["end"] = int(a), int(b)
        except ValueError:
            pass
    if len(parts) >= 3:
        try:
            out["single"] = int(parts[2])
        except ValueError:
            pass
    return out


def live_cvec_matches(live, desired) -> bool:
    """Совпадает ли состояние движка с желаемым (mode - всегда; слои - когда заданы
    с обеих сторон; single-слой - когда задан с обеих)."""
    if not live or not desired:
        return False
    if str(live.get("mode")) != str(desired.get("mode")):
        return False
    for k in ("start", "end", "single"):
        a, b = live.get(k), desired.get(k)
        if a is not None and b is not None and int(a) != int(b):
            return False
    return True


# ------------------------------------------------------------------ скан векторов
_SCAN_CACHE: dict = {}          # (корни) -> (monotonic, {показ: путь})
_SCAN_TTL = 30.0
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".cache", "logs"}
MAX_VECTOR_BYTES = 256 * 1024 * 1024      # векторы ~0.5 МБ; модели отсекаем по размеру


def scan_control_vectors(roots, ttl: float = _SCAN_TTL, force: bool = False):
    """Найти GGUF-контрольные векторы под корнями: {показ: абсолютный путь}.
    Показ - путь относительно своего корня (прямой слэш). Вектор = GGUF, в котором
    есть тензоры direction.<N>; большие файлы (>256 МБ) не вскрываются вовсе.
    Кеш TTL (сек), чтобы /object_info не сканировал диск на каждый F5."""
    roots = tuple(str(r) for r in roots if str(r).strip())
    if not roots:
        return {}
    now = time.monotonic()
    key = tuple(sorted(set(roots)))
    if not force:
        hit = _SCAN_CACHE.get(key)
        if hit and now - hit[0] < ttl:
            return dict(hit[1])
    found: dict = {}
    for root in key:
        root_p = Path(root)
        if not root_p.is_dir():
            continue
        base_depth = len(root_p.parts)
        for dirpath, dirnames, filenames in os.walk(root_p):
            dp = Path(dirpath)
            if len(dp.parts) - base_depth >= 4:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                if not fn.lower().endswith(".gguf"):
                    continue
                fp = dp / fn
                try:
                    if fp.stat().st_size > MAX_VECTOR_BYTES or fp.stat().st_size < 256:
                        continue
                    layers = control_vector_layers(fp)
                except OSError:
                    continue
                if layers:
                    show = fp.relative_to(root_p).as_posix()
                    found[show] = str(fp)
    _SCAN_CACHE[key] = (now, dict(found))
    return dict(found)


def env_roots() -> list:
    """Доп. корни из переменной окружения POLYSTRATA_CVEC_DIR (разделитель -
    os.pathsep: ';' на Windows, ':' на POSIX)."""
    raw = os.environ.get("POLYSTRATA_CVEC_DIR", "")
    return [p.strip() for p in raw.split(os.pathsep) if p.strip()]


def resolve_vector_path(value, roots):
    """Показ из дропдауна (путь относительно корня) или абсолютный путь -> Path
    или None. Ищет по корням по порядку, затем как есть."""
    v = str(value or "").strip()
    if not v:
        return None
    cand = Path(v)
    if cand.is_absolute() and cand.is_file():
        return cand
    for r in roots:
        p = Path(r) / v
        if p.is_file():
            return p
    return cand if cand.is_file() else None


def parse_extra_vectors(text, roots):
    """Доп. векторы из multiline-строки: «путь:скейл» по одному в строке (или через
    запятую). Путь - абсолютный или относительно корней; скейл отделён последним
    «:», если справа число (пути Windows с «D:\» не ломаются). -> ([(Path, float)],
    [строки-предупреждения])."""
    entries, warns = [], []
    for chunk in re.split(r"[\n,]+", str(text or "")):
        chunk = chunk.strip()
        if not chunk:
            continue
        path_part, scale = chunk, 1.0
        if ":" in chunk:
            head, tail = chunk.rsplit(":", 1)
            try:
                scale = float(tail)
                path_part = head
            except ValueError:
                pass                                  # «:» - часть пути/не число
        p = resolve_vector_path(path_part, roots)
        if p is None:
            warns.append(f"доп. вектор не найден: {chunk}")
            continue
        entries.append((p, scale))
    return entries, warns


# ------------------------------------------------------------------ вариант-конфиг
def sanitize_suffix(suffix: str) -> str:
    """«cvec» / «Soft 2!» -> «cvec» / «Soft2»: только [A-Za-z0-9._-], без краёв-пробелов."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(suffix or "").strip())
    return s.strip("-_.") or "cvec"


def variant_paths(base_config: Path, suffix: str):
    """(путь варианта, id варианта) по конфигу базы:
    strata-<stem>.json + «-cvec» -> strata-<stem>-cvec.json, id берётся из model_name."""
    base_config = Path(base_config)
    sfx = sanitize_suffix(suffix)
    return base_config.with_name(base_config.stem + "-" + sfx + ".json"), sfx


def variant_config(base_cfg: dict, base_id: str, cvec_args, suffix: str,
                   requests_default: bool):
    """Клон базового конфига + cvec-флаги (старые cvec-флаги базы вычищены):
    свой model_name/display_name/лог, без aliases; sampling.experimental_speed_projection
    = умолчание для запросов, не указавших поле. Неизвестные серверу ключи
    (маркер cvec_variant) безвредны - см. докстринг модуля."""
    cfg = json.loads(json.dumps(base_cfg))          # глубокая копия
    sfx = sanitize_suffix(suffix)
    args = strip_cvec_args(cfg.get("args"))
    cfg["args"] = args + [str(a) for a in cvec_args]
    cfg["model_name"] = f"{base_id}-{sfx}"
    base_disp = str(cfg.get("display_name") or base_id).rstrip()
    cfg["display_name"] = f"{base_disp} + {sfx}"
    cfg.pop("aliases", None)                        # не отвечать на имена базы
    log = cfg.get("log")
    if log:
        lp = Path(str(log))
        cfg["log"] = str(lp.with_name(lp.stem + "-" + sfx + lp.suffix))
    s = dict(cfg.get("sampling")) if isinstance(cfg.get("sampling"), dict) else {}
    s["experimental_speed_projection"] = bool(requests_default)
    cfg["sampling"] = s
    cfg[CVEC_MARKER] = {"of": base_id, "suffix": sfx, "by": "ComfyUI-PolyStrata"}
    return cfg


def is_our_variant(cfg: dict) -> bool:
    """Конфиг создан этой нодой (есть маркер)?"""
    return isinstance(cfg, dict) and isinstance(cfg.get(CVEC_MARKER), dict)


def write_config_if_changed(path: Path, cfg: dict) -> bool:
    """Атомарная запись (tmp + os.replace, как serve/runconfig.py.save).
    True - файл записан; False - содержимое не изменилось (перезапись не нужна)."""
    path = Path(path)
    text = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
    try:
        if path.read_text(encoding="utf-8-sig") == text:
            return False
    except OSError:
        pass
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return True


def load_config(path):
    """JSON-конфиг по пути (utf-8-sig - как читает сервер); ValueError - битый."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
