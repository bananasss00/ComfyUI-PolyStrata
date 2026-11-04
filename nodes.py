"""nodes.py - ComfyUI-ноды для сервера PolyStrata (https://github.com/VecSzn/PolyStrata).
Архитектура v4: сервер запускает ПОЛЬЗОВАТЕЛЬ, ноды только шлют запросы по HTTP.

Ноды:
  PolyStrata Chat     - генерация текста (без стриминга, ответ целиком). SCHEMA V3
                        (comfy_api.latest): ДИНАМИЧЕСКИЕ ВХОДЫ - autogrow-пины картинок
                        (img_0, img_1, ...) и ОТДЕЛЬНО видео (vid_0, vid_1, ...), как в
                        llama.cpp-нодах (LlamaCPP-Subprocess). Каждый vid-пин = отдельное
                        видео: блок "[Video N]" + кадры по порядку, при заданном video_fps -
                        метки времени '<X.X seconds>' (формат llama.cpp при декоде видео).
                        reasoning/temperature/top_p/seed - только здесь.
                        Выходы: text + thoughts (размышления отдельно).
                        Лимиты max_images/max_video_frames - НА КАЖДЫЙ пин своего типа;
                        перед отправкой кадры подгоняются под контекст движка (двигок сам
                        запросы НЕ обрезает - отклоняет 400 'exceeds the context window').
                        Тоггл clear_comfy_vram: перед генерацией выгрузить модели САМОГО
                        ComfyUI (unload_all_models + torch-кэш, как кнопка 'Free model and
                        node cache'); в одном запуске воркфлоу - ОДИН РАЗ (листы/циклы/сиды
                        не дёргают память повторно, флаг сбрасывает хук execute). Листы на
                        любых входах (list-паки) обрабатываются безопасно и БЕЗ хардкода
                        имён: _scalars прогоняет весь kwargs на входе ноды - скалярные
                        виджеты берут первый элемент (новые виджеты защищены автоматически),
                        списки картинок/видео разворачиваются в пины (_as_pins).
                        Interrupt (кнопка Cancel ComfyUI) прерывает генерацию: соединение
                        закрывается, сервер отменяет запрос при обрыве клиента.
  PolyStrata Sampling - тонкий сэмплинг БЕЗ дубликатов (temperature/top_p/reasoning/seed живут
                        в ноде Chat): top_k, min_p, штрафы, stop + logit_bias
                        (запрет токенов, как banned_tokens в llama-нодах) + extra_sampling
                        (JSON-сквозняк: любые поля запроса - задел на будущие семплеры).
  PolyStrata Memory   - обработчик истории (хендлер): копит пары вопрос-ответ между
                        запусками. Без него Chat работает одиночным запросом.
  PolyStrata Unload   - полная выгрузка модели (POST /v1/unload). Сквозная. Возвращает
                        ComfyUI исходное EXTRA_RESERVED_VRAM (бронь снимается).

Резерв VRAM для ComfyUI (reserve_mib/extra_mib в Chat) - логика update_global_vram_reservation
из llama.cpp-нод (LlamaCPP-Subprocess): фактическое потребление PolyStrata (total GPU -
фактически свободно) + extra_mib записывается в comfy.model_management.EXTRA_RESERVED_VRAM
и в headroom DynamicVRAM (comfy-aimdo), поэтому ComfyUI грузит свои модели, не выходя за
пределы реального окна, и не ловит OOM. Unload возвращает исходные значения.

Семплеры, которых у PolyStrata НЕТ (проверено по serve/server.py текущего main:
check_request_sampling + StrataEngine.sampling_keys - единственный белый список):
xtc, dry, dynatemp, mirostat, tfs_z, typical_p, smoothing_factor/curve, top_n_sigma.
Движок strata-poly - не ванильный llama.cpp, этих сэмплеров в нём нет, сервер молча
отбрасывает такие поля. Если появятся - их можно передать без обновления нод через
виджет extra_sampling ноды Sampling.

ВАЖНО (elastic VRAM): сегментированный кэш экспертов ВСТРОЕН в движок strata-poly
(0.1.45-poly.6 и новее) - ключи в конфиг модели (strata-<модель>.json) добавлять НЕ нужно.
Ключ "vram_elastic": true в конфиге ВРЕДЕН: сервер передаст движку неизвестный флаг
--vram-elastic и движок не стартует ("unknown option"). Если ключ попал в конфиг - удалите
его и запустите модель заново.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .ps_client import PolyStrataClient, PolyStrataError
except ImportError:                      # загрузка не как пакет (старый загрузчик ComfyUI)
    from ps_client import PolyStrataClient, PolyStrataError

# ---------------------------------------------------------------------------
# Schema V3 (comfy_api.latest) - как в llama.cpp-нодах. Если ComfyUI старый и
# comfy_api нет - Chat собирается в V1 (фиксированные входы images/videos).
try:
    from comfy_api.latest import io as comfy_io     # алиас: не затеняем стандартный io
    _HAS_V3 = True
except Exception:
    comfy_io = None
    _HAS_V3 = False

# ---------------------------------------------------------------- утилиты типов
class AnyType(str):
    """Тип, совпадающий с любым другим (сквозные входы/выходы any_input/any_output)."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")
PS_MEMORY = "PS_MEMORY"        # выход ноды Memory -> вход 'memory' ноды Chat
PS_SAMPLING = "PS_SAMPLING"    # выход ноды Sampling -> вход 'sampling' ноды Chat

# ---------------------------------------------------------------- interrupt (Cancel ComfyUI)
try:
    # ВАЖНО: в ComfyUI этот класс наследует BaseException (не Exception) - его не глотают
    # широкие except Exception в пользовательском коде, и мы его тоже не глотаем.
    from comfy.model_management import InterruptProcessingException as _Interrupted
except Exception:
    class _Interrupted(BaseException):
        """Заглушка (comfy.model_management недоступен): прерывание работать не будет."""


def _throw_if_interrupted() -> None:
    """Кнопка Cancel (interrupt) в ComfyUI: comfy.model_management.
    throw_exception_if_processing_interrupted() бросает InterruptProcessingException, если
    пользователь прервал выполнение. Вызывается до запроса, между шагами и ПОЛЛИНГОМ во время
    генерации (ps_client при срабатывании закрывает соединение - сервер PolyStrata/llama.cpp
    отменяет генерацию при обрыве клиента). Исключение намеренно НЕ ловится здесь:
    его обрабатывает сам ComfyUI (останавливает выполнение)."""
    try:
        import comfy.model_management as mm
    except Exception:
        return
    fn = getattr(mm, "throw_exception_if_processing_interrupted", None)
    if fn is not None:
        fn()

# ================================================================ пресеты системных промптов
# ТАК ЖЕ, как в llama.cpp-нодах (ComfyUI-LlamaCPP-Subprocess): папка
# ComfyUI/models/LLM/prompts, один пресет = один .txt файл (можно в подкаталогах,
# в дропдауне виден относительный путь "subdir/prompt.txt").
# Обновление списка: файлы сканируются КАЖДЫЙ РАЗ при построении ноды (define_schema/
# INPUT_TYPES) - БЕЗ кеша folder_paths.get_filename_list(), который не инвалидируется при
# ручном добавлении файлов. При обновлении вкладки браузера ComfyUI перечитывает
# /object_info - ноды строятся заново, и дропдауны показывают актуальный список файлов.
NO_SYSTEM_PROMPT = "none"
PROMPT_FOLDER = "llm_text_processor_prompts"   # то же имя папки, что в llama.cpp-нодах

# Стандартные пресеты - те же файлы, что создаёт llama.cpp-пак; каждый недостающий
# файл досоздаётся при загрузке (если llama.cpp-ноды стоят рядом - всё уже существует).
DEFAULT_PROMPTS = {
    "prompt_refine_expand.txt": "Refine and enhance the following user prompt for creative text-to-image generation. Keep the meaning and keywords, make it more expressive and visually rich. Output ONLY the improved prompt text (no preface, no bullets, no JSON, no <think>, no commentary).",
    "short_story.txt": "Write a short, imaginative story inspired by this image or video.",
    "video_summary.txt": "Summarize the key events and narrative points in this video.",
    "detailed_analysis.txt": "Output ONLY these sections with short labels (no bullets): Subject; People (if any); Environment; Lighting; Camera/Composition; Color/Texture. In each section, write 2–4 sentences of concrete visible details. If something is not visible, write 'not visible'. No preface, no reasoning, no <think>.",
    "cinematic_description.txt": "Write ONE cinematic paragraph (8–12 sentences). Describe the scene like a film still: subject(s) and action; environment and atmosphere; lighting design (practical lights vs ambient, direction, contrast); camera language (shot type, angle, lens feel, depth of field, motion implied); composition and mood. Keep it vivid but factual (no made-up story). No preface, no reasoning, no <think>.",
    "ultra_detailed_description.txt": "Write ONE ultra-detailed paragraph (10–16 sentences, ~180–320 words). Stay grounded in visible details. Include: subject micro-details (materials, textures, patterns, wear, reflections); people details if present (hair, skin tones, makeup, jewelry, fabric types, fit); environment depth (foreground/midground/background, signage/props, surface materials); lighting analysis (key/fill/back light, direction, softness, highlights, shadow shape); camera perspective (angle, lens feel, depth of field) and composition (leading lines, negative space, symmetry/asymmetry, visual hierarchy). No preface, no reasoning, no <think>.",
    "detailed_description.txt": "Write ONE detailed paragraph (6–10 sentences). Describe only what is visible: subject(s) and actions; people details if present (approx age group, gender expression if clear, hair, facial expression, pose, clothing, accessories); environment (location type, background elements, time cues); lighting (source, direction, softness/hardness, color temperature, shadows); camera viewpoint (eye-level/low/high, distance) and composition (framing, focal emphasis). No preface, no reasoning, no <think>.",
    "simple_description.txt": "Analyze the image and write a single concise sentence that describes the main subject and setting. Keep it grounded in visible details only.",
    "tags.txt": "Your task is to generate a clean list of comma-separated tags for a text-to-image AI, based *only* on the visual information in the image. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags."
}


def prompt_root() -> Path:
    """Папка пресетов - ТОТ ЖЕ путь, что в llama.cpp-нодах: models/LLM/prompts."""
    try:
        import folder_paths
        return Path(folder_paths.models_dir) / "LLM" / "prompts"
    except Exception:
        return Path(os.getcwd()) / "models" / "LLM" / "prompts"


def _register_prompt_folder() -> None:
    """Создать папку пресетов, недостающие стандартные пресеты и зарегистрировать
    папку в folder_paths (то же имя и содержимое, что у llama.cpp-нод)."""
    try:
        d = prompt_root()
        d.mkdir(parents=True, exist_ok=True)
        for filename, content in DEFAULT_PROMPTS.items():
            fp = d / filename
            if not fp.exists():
                try:
                    fp.write_text(content, encoding="utf-8")
                    print(f"[PolyStrata] создан стандартный пресет: {filename}", flush=True)
                except Exception as e:
                    print(f"[PolyStrata] предупреждение: не удалось создать пресет "
                          f"{filename}. Ошибка: {e}", flush=True)
        import folder_paths
        folder_paths.folder_names_and_paths[PROMPT_FOLDER] = ([str(d)], {".txt"})
    except Exception as e:
        print(f"[PolyStrata] папка пресетов системных промптов недоступна ({e}) - "
              f"список пресетов будет пуст (только 'none')", flush=True)


def system_prompt_options() -> list:
    """Свежее сканирование папки пресетов при КАЖДОМ вызове, минуя кеш folder_paths
    (get_filename_list кеширует и не инвалидируется при ручном добавлении файлов -
    из-за этого список не обновлялся бы при перезагрузке страницы ComfyUI).
    Рекурсивно: в дропдауне - относительные пути ("subdir/my_prompt.txt")."""
    d = prompt_root()
    if not d.exists():
        return [NO_SYSTEM_PROMPT]
    try:
        files = sorted(
            str(f.relative_to(d)).replace(os.sep, "/")
            for f in d.rglob("*.txt") if f.is_file()
        )
    except Exception:
        files = []
    return [NO_SYSTEM_PROMPT] + files


def resolve_system_prompt(preset_name: str, manual_text: str) -> str:
    """Логика llama.cpp-нод: выбран пресет (не 'none') - используется ТОЛЬКО он,
    ручной текст игнорируется; 'none' - берётся ручной system_prompt_text."""
    if preset_name and preset_name != NO_SYSTEM_PROMPT:
        p = prompt_root() / preset_name
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8").strip()
            except Exception as e:
                print(f"[PolyStrata] пресет '{preset_name}' не прочитан: {e}", flush=True)
                return ""
        print(f"[PolyStrata] пресет '{preset_name}' не найден в {prompt_root()} - "
              f"системный промпт пуст", flush=True)
        return ""
    return (manual_text or "").strip()


_register_prompt_folder()


MAX_PIN_SLOTS = 50             # autogrow: максимум пинов в группе img_* / vid_*
MAX_IMAGES_PER_PIN = 64        # потолок кадров одного image-пина (до выборки)
MAX_VIDEO_FRAMES_PER_PIN = 128 # потолок кадров одного видео-пина (до выборки)
JPEG_QUALITY = 90
DEFAULT_FRAME_SIDE = 1024      # длинная сторона кадра после ужатия (виджет max_frame_side)
TOKEN_PX = 28                  # у Qwen-VL один vision-токен ~ 28x28 пикселей (оценка с запасом:
                               # реальные 32 у Qwen3-VL - оценка выше факта на ~30%)
FRAME_OVERHEAD_TOKENS = 12     # спецтокены кадра в промпте ([Image K]/[Video N]/'<X.X seconds>')
TEXT_CHARS_PER_TOKEN = 3.5     # оценка токенов текста по символам
TEXT_OVERHEAD_TOKENS = 128     # запас на шаблон чата и служебные токены
MEDIA_WARN_FRAMES = 100        # предупреждение, если суммарно кадров больше

SERVER_TOOLTIP = (
    "IP:порт сервера PolyStrata, например 127.0.0.1:8080 (порт можно опустить - подставится 8080).\n"
    "Сервер вы запускаете САМИ (run-<модель>.bat / START-HERE.bat); ноды только шлют запросы."
)

API_KEY_TOOLTIP = (
    "API-ключ, если на сервере PolyStrata задан 'api_key' (нужен при доступе не с localhost). "
    "Пусто - не отправлять."
)

ELASTIC_NOTE = (
    "Elastic VRAM у strata-poly встроен (движок 0.1.45-poly.6 и новее, обновление - UPDATE.bat): "
    "ключи в конфиг модели не нужны, и ключ \"vram_elastic\": true там ВРЕДЕН (движок падает на "
    "неизвестном флаге --vram-elastic). Одна NVIDIA (CUDA VMM)."
)

RESERVE_TOOLTIP = (
    "Сколько MiB VRAM держать СВОБОДНЫМИ для ComfyUI (POST /v1/vram, elastic): модель на "
    "пониженном потреблении - кэш экспертов выгружается сегментами по 512 MiB, пока свободной "
    "не станет столько; недостающие эксперты считаются на CPU (декод медленнее).\n"
    "0 = НЕ ВЛИЯЕТ ни на PolyStrata, ни на бронь ComfyUI (никаких запросов).\n"
    "Значение сервер запоминает до сброса (Unload) или перезапуска сервера. Если экспертов не "
    "хватило, будет частичное освобождение (см. лог). " + ELASTIC_NOTE
)

EXTRA_TOOLTIP = (
    "Дополнительный запас MiB. Два применения:\n"
    "1) цель свободной VRAM на сервере = reserve_mib + extra_mib;\n"
    "2) в бронь ComfyUI (EXTRA_RESERVED_VRAM, как в llama.cpp-нодах) пишется ФАКТИЧЕСКОЕ "
    "потребление PolyStrata (сколько не удалось освободить) + extra_mib - запас под "
    "prompt-path loan движка на время генерации, чтобы ComfyUI не ловил OOM."
)

CLEAR_VRAM_TOOLTIP = (
    "Тоггл: ПЕРЕД генерацией выгрузить модели САМОГО ComfyUI (checkpoint/LoRA/ControlNet и "
    "пр.) и почистить torch-кэш - то же, что кнопка 'Free model and node cache': VRAM "
    "освобождается под запрос PolyStrata; при нужде ComfyUI загрузит модели обратно сам. "
    "Модель PolyStrata это НЕ задевает.\n"
    "В ОДНОМ запуске воркфлоу очистка делается ОДИН РАЗ: лист на входе (цикл по картинкам, "
    "батчи), перебор сидов, несколько нод Chat - нода исполняется многократно, но после "
    "первой очистки выгружать уже нечего (результат тот же), поэтому повторные пропускаются. "
    "С началом следующего запуска (очередь) очистка снова доступна.\n"
    "Выкл (по умолчанию) - нода не трогает память ComfyUI."
)

IMAGES_TOOLTIP = (
    "Картинки (IMAGE). Autogrow: подключите img_0 - появится img_1, и так до 50 пинов. "
    "Каждый пин - картинка ИЛИ батч [N,H,W,C]: кадры батча раскрываются по одному с метками "
    "[Image 1], [Image 2], ... (сквозная нумерация). Нужен vision (mmproj) в конфиге модели."
)

VIDEOS_TOOLTIP = (
    "Видео как батч последовательных кадров (IMAGE [N,H,W,C] - выход 'Load Video'/Video Helper "
    "Suite). Autogrow: vid_0, vid_1, ... до 50 пинов. КАЖДЫЙ пин = ОТДЕЛЬНОЕ видео: блок "
    "'[Video N]' и кадры по порядку; при заданном video_fps между кадрами вставляются метки "
    "времени '<X.X seconds>' - так же llama.cpp размечает кадры при декодировании видео. "
    "(Jan шлёт видеофайл целиком - input_video, и движок сам нарезает кадры с теми же метками; "
    "PolyStrata input_video не принимает, поэтому кадры размечаем на ноде.) Без меток времени "
    "модель часто описывает видео ПО КАДРАМ ('Frame 1: ...', 'Frame 2: ...'). Аудио не передаётся."
)

MAX_IMAGES_TOOLTIP = (
    "Лимит кадров на КАЖДЫЙ пин картинок (внутри пина выборка равномерная - np.linspace): "
    "0 = все кадры пина (потолок 64/пин). Каждый кадр ест контекст vision: "
    "~ (высота/28)*(ширина/28) токенов у Qwen-VL, и двигок запросы НЕ обрезает - отклоняет целиком."
)

MAX_VIDEO_FRAMES_TOOLTIP = (
    "Лимит кадров на КАЖДЫЙ видео-пин (внутри пина выборка равномерная - np.linspace): "
    "два видео и 8 = по 8 кадров на КАЖДОЕ. 0 = все кадры пина (потолок 128/пин). Каждый кадр - "
    "~ (высота/28)*(ширина/28) токенов контекста (кадр 1280x720 ~ 1200 токенов), двигок запросы "
    "НЕ обрезает; под окно движка кадры режутся заранее (см. лог). Метки времени между "
    "кадрами - виджет video_fps."
)

VIDEO_FPS_TOOLTIP = (
    "FPS исходного видео - для меток времени между кадрами ('<0.5 seconds>' - формат llama.cpp "
    "при декоде видео). Метки дают модели темпоральность: она описывает видео как непрерывное "
    "событие, а не перечисляет кадры по одному. 0 = без меток (только блок '[Video N]'). "
    "Типичные значения 16-30. Время считается от индексов кадров ИСХОДНОГО батча (выборка "
    "max_video_frames сохраняет исходные индексы), поэтому подставляйте fps именно исходника."
)

FRAME_SIDE_TOOLTIP = (
    "Длинная сторона кадра после ужатия перед отправкой (картинки и кадры видео): чем меньше, "
    "тем меньше vision-токенов ест кадр. 1280x720 -> 1024x576 ~ 780 токенов, -> 640x360 ~ 320. "
    "0 = без ужатия (не рекомендуется). По умолчанию 1024."
)

SEED_TOOLTIP = (
    "Seed сэмплинга. 0 = случайный (поле не отправляется); >0 - фиксированный. "
    "control_after_generate работает (fix/increment/decrement/randomize).\nВнимание (PolyStrata): "
    "с адаптивным resident-уровнем экспертов повтор с тем же seed не обязан давать тот же текст."
)

SAMPLERS_NOTE = (
    "Семплеров xtc/dry/dynatemp/mirostat/tfs_z/typical_p/smoothing/top_n_sigma у PolyStrata НЕТ "
    "(движок strata-poly - не ванильный llama.cpp; проверено по serve/server.py). Сервер молча "
    "отбрасывает такие поля. Если появятся в новом движке - передайте их через extra_sampling, "
    "не дожидаясь обновления нод."
)

LOGIT_BIAS_TOOLTIP = (
    "Запрет/подталкивание токенов (logit_bias), как banned_tokens в llama.cpp-нодах. Строка на "
    "токен: '12345' - ЗАПРЕТИТЬ токен 12345 (bias -100); '12345:-2.5' или '12345=-2.5' - bias "
    "-2.5 (диапазон -100..100). Список токенов модели - в логе сервера или /v1/status. "
    "# - комментарий."
)

EXTRA_SAMPLING_TOOLTIP = (
    "JSON-объект с ЛЮБЫМИ дополнительными полями запроса (сквозняк на будущее): "
    '{"top_n_sigma": 1.0, "xtc_probability": 0.5}. Сливается в тело запроса как есть - '
    "если PolyStrata добавит новые семплеры, они заработают без обновления нод. Сегодня "
    "неизвестные поля сервер молча игнорирует. Неверный JSON - ошибка ноды."
)


# ---------------------------------------------------------------- кадры -> base64
def _scaled_dims(h: int, w: int, max_side: int) -> tuple:
    """Размеры кадра после ужатия длинной стороны до max_side (max_side <= 0 - без ужатия)."""
    if max_side and max_side > 0:
        long_side = max(h, w)
        if long_side > max_side:
            s = max_side / float(long_side)
            return max(1, round(h * s)), max(1, round(w * s))
    return h, w


def _frame_tokens(h: int, w: int, max_side: int) -> int:
    """~токенов контекста на кадр у Qwen-VL после ужатия: (H/28)*(W/28) + спецтокены."""
    th, tw = _scaled_dims(h, w, max_side)
    return ((th + TOKEN_PX - 1) // TOKEN_PX) * ((tw + TOKEN_PX - 1) // TOKEN_PX) + FRAME_OVERHEAD_TOKENS


def _pin_array(images) -> np.ndarray:
    """Пин (IMAGE [H,W,C] или [N,H,W,C]) -> numpy uint8 RGB [N,H,W,C]
    (float 0..1 из ComfyUI, torch-тензор на GPU, RGBA/серый - всё приводится)."""
    if hasattr(images, "cpu"):
        images = images.cpu().numpy()
    arr = np.asarray(images)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4 or arr.shape[-1] not in (1, 3, 4):
        raise PolyStrataError(f"Неожиданная форма {arr.shape}, ожидаю [N,H,W,C] с 1/3/4 каналами")
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)   # IMAGE в ComfyUI всегда float 0..1
    if arr.shape[-1] == 4:                        # RGBA -> RGB
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _encode_jpeg_b64(arr: np.ndarray, max_side: int = DEFAULT_FRAME_SIDE) -> str:
    """Кадр uint8 [H,W,C] -> data:image/jpeg;base64,... (длинная сторона ужата до max_side)."""
    h, w = arr.shape[:2]
    th, tw = _scaled_dims(h, w, max_side)
    if (th, tw) != (h, w):
        arr = np.asarray(Image.fromarray(arr).resize((tw, th), Image.LANCZOS))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _sample_indices(total: int, count: int) -> list:
    """count индексов, равномерно покрывающих total кадров (np.linspace, без повторов)."""
    count = max(1, min(int(count), total))
    if count == total:
        return list(range(total))
    idx = np.round(np.linspace(0, total - 1, count)).astype(int)
    seen, out = set(), []
    for i in idx.tolist():
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _pin_allowance(count: int, limit: int, hard_cap: int) -> int:
    """Сколько кадров взять из ОДНОГО пина: limit - лимит НА ПИН (0 = без лимита, до hard_cap)."""
    if limit and int(limit) > 0:
        return max(0, min(int(count), int(limit), hard_cap))
    return max(0, min(int(count), hard_cap))


def _sorted_autogrow(d) -> list:
    """Autogrow-вход приходит словарём {'img_0': ..., 'img_1': ...} - сортируем по номеру
    пина (img_0, img_1, ...), чтобы порядок совпадал с порядком подключения."""
    if not d:
        return []

    def _key(k):
        if "_" in k:
            _, _, n = k.rpartition("_")
            if n.isdigit():
                return (0, int(n))
        return (1, str(k))

    return [d[k] for k in sorted(d.keys(), key=_key)]


def _scalar(value, name: str):
    """Значение виджета пришло СПИСКОМ (list-паки циклов/переборов шлют листы в любые входы,
    не только в картинки) -> берётся ПЕРВЫЙ элемент, чтобы нода не упала и отработала
    предсказуемо: скалярные виджеты (prompt/seed/резервы/...) не умеют листы, а результат от
    этого не меняется - очистка памяти и запрос делаются один раз (см. _maybe_clear_comfy_vram).
    Пустой список -> None (виджет считается незаданным)."""
    if isinstance(value, (list, tuple)):
        v = value[0] if len(value) else None
        print(f"[PolyStrata] {name}: пришёл список ({len(value)} знач.) - взят первый элемент",
              flush=True)
        return v
    return value


# Единственные входы, которым списки ЛЕГАЛЬНЫ: autogrow-пины/батчи картинок и видео -
# их разворачивает в пины _as_pins (рекурсивно, списки списков включительно).
_LIST_INPUTS = ("images", "videos")


def _scalars(kwargs: dict) -> dict:
    """Универсальная нормализация входов ноды БЕЗ хардкода имён виджетов: весь kwargs
    прогоняется через _scalar - ЛЮБОЙ вход, пришедший списком (list-паки циклов/переборов,
    батчи сидов, листы в саб-ноды), заменяется первым элементом. Имена не перечисляются,
    поэтому новый виджет защищён автоматически - при добавлении поля не нужно помнить про
    листы. Исключение только семантическое - _LIST_INPUTS (images/videos): там список и
    есть значение, его разворачивает в пины _as_pins."""
    return {k: v if k in _LIST_INPUTS else _scalar(v, k) for k, v in kwargs.items()}


def _as_pins(v) -> list:
    """V3 autogrow dict {'img_0': ...} -> список значений по номерам пинов; одиночный вход
    (тензор/массив IMAGE из V1-фолбэка или одиночного подключения) -> [значение].
    Список от list-паков (['батч', 'батч'] или список одиночных картинок) -> элементы
    по порядку: каждый батч остаётся отдельным пином, кадры нумеруются сквозняком."""
    if v is None:
        return []
    if isinstance(v, dict):
        return _sorted_autogrow(v)
    if isinstance(v, (list, tuple)):
        out: list = []
        for item in v:
            out.extend(_as_pins(item))
        return out
    return [v]


def _plan_media(images, videos, max_images: int, max_video_frames: int) -> list:
    """План медиа-групп по autogrow-пинам:
    [{'kind': 'image'|'video', 'num': N, 'arr': [N,H,W,C], 'idx': [индексы кадров], '_orig': n}].
    Лимиты max_images / max_video_frames - НА КАЖДЫЙ пин своего типа (0 = все кадры пина,
    потолки 64/128 кадров на пин сохраняются). Общий объём страхует префлайт контекста
    (_fit_media_to_context), а не делёжка бюджета между пинами."""
    groups = []
    for arr in (_pin_array(p) for p in _as_pins(images) if p is not None):
        n_total = int(arr.shape[0])
        al = _pin_allowance(n_total, max_images, MAX_IMAGES_PER_PIN)
        if al < n_total:
            print(f"[PolyStrata] images: пин с {n_total} кадрами -> {al} "
                  f"(лимит max_images НА ПИН)", flush=True)
        groups.append({"kind": "image", "num": 0, "arr": arr,
                       "idx": _sample_indices(n_total, al), "_orig": n_total})
    for i, arr in enumerate((_pin_array(p) for p in _as_pins(videos) if p is not None)):
        n_total = int(arr.shape[0])
        al = _pin_allowance(n_total, max_video_frames, MAX_VIDEO_FRAMES_PER_PIN)
        if al < n_total:
            print(f"[PolyStrata] videos: видео {i + 1}: {n_total} кадров -> {al} "
                  f"(лимит max_video_frames НА ПИН)", flush=True)
        groups.append({"kind": "video", "num": i + 1, "arr": arr,
                       "idx": _sample_indices(n_total, al), "_orig": n_total})
    return groups


def _media_tokens(groups: list, max_side: int) -> int:
    """Оценка vision-токенов всех кадров плана (после ужатия до max_side)."""
    tot = 0
    for g in groups:
        tot += _frame_tokens(int(g["arr"].shape[1]), int(g["arr"].shape[2]), max_side) * len(g["idx"])
    return tot


def _fit_media_to_context(groups: list, n_ctx: int, max_tokens: int, text_tokens: int,
                          max_side: int) -> tuple:
    """Подгонка числа кадров под контекст движка (n_ctx): двигок запрос НЕ обрезает - отклоняет
    целиком (400 'exceeds the context window'), поэтому кадры урезаются ДО отправки.
    Оценка кадра консервативная (28 px/токен; у Qwen3-VL фактически 32) - в окно влезем и по факту.
    Возвращает (группы, заметка-строка или None)."""
    if not n_ctx or int(n_ctx) <= 0 or not groups:
        return groups, None
    headroom = int(max_tokens) if max_tokens and int(max_tokens) > 0 else 512
    room = int(n_ctx) - int(text_tokens) - TEXT_OVERHEAD_TOKENS - headroom
    was = sum(len(g["idx"]) for g in groups)
    if was == 0:
        return groups, None
    for _ in range(8):
        est = _media_tokens(groups, max_side)
        total = sum(len(g["idx"]) for g in groups)
        if total == 0 or est <= room:
            break
        if room <= 0:                        # текст уже занимает всё окно - кадры не помогут
            for g in groups:
                g["idx"] = []
            break
        ratio = room / float(est)
        changed = False
        for g in groups:                     # режем пропорционально; кого округло не срезало - вдвое
            new_n = int(len(g["idx"]) * ratio)
            if new_n >= len(g["idx"]):
                new_n = len(g["idx"]) // 2
            if new_n < len(g["idx"]):
                changed = True
            g["idx"] = _sample_indices(g["_orig"], new_n) if new_n > 0 else []
        if not changed:
            for g in groups:
                g["idx"] = []
            break
    now = sum(len(g["idx"]) for g in groups)
    if now == was:
        return groups, None
    note = (f"КОНТЕКСТ {int(n_ctx)} токенов: кадров {was} -> {now} (подгонка под окно движка; "
            f"окончательно управляют max_video_frames/max_images/max_frame_side)")
    return groups, note


def _video_time_marker(idx: int, fps: float) -> str:
    """Метка времени кадра в формате llama.cpp при декодировании видео (mtmd-helper.cpp:
    snprintf(ts_buf, "<%.1f seconds>", ts_s) перед каждой группой кадров): '<0.5 seconds>'.
    Это якорь темпоральности: модель trained на видео с такими метками видит одну
    последовательность, а не N отдельных картинок."""
    return f"<{idx / float(fps):.1f} seconds>"


def _build_media_parts(groups: list, max_side: int, prompt: str, video_fps: int = 0) -> tuple:
    """План -> OpenAI content-части: текст промпта + метки и base64-кадры.
    Картинки - сквозные метки [Image 1], [Image 2], ...; каждый видео-пин - блок '[Video N]'
    с кадрами по порядку. При заданном video_fps перед КАЖДЫМ кадром вставляется метка
    '<X.X seconds>' (формат llama.cpp: Jan шлёт видеофайл - движок сам нарезает кадры и
    вставляет те же метки; PolyStrata input_video не принимает, размечаем на ноде).
    Без меток модель легко отвечает по кадрам ('Frame 1: ...', 'Frame 2: ...')."""
    parts: list = [{"type": "text", "text": prompt}]
    total = 0
    fps = float(video_fps or 0)
    for g in groups:
        if g["kind"] == "image":
            for i in g["idx"]:
                total += 1
                parts.append({"type": "text", "text": f"[Image {total}]"})
                parts.append({"type": "image_url",
                              "image_url": {"url": _encode_jpeg_b64(g["arr"][i], max_side)}})
        elif g["idx"]:
            total += len(g["idx"])
            parts.append({"type": "text", "text": f"[Video {g['num']}]"})
            for i in g["idx"]:                 # i - индекс в ИСХОДНОМ батче кадров
                if fps > 0:
                    parts.append({"type": "text", "text": _video_time_marker(i, fps)})
                parts.append({"type": "image_url",
                              "image_url": {"url": _encode_jpeg_b64(g["arr"][i], max_side)}})
    return parts, total


def _collect_media_parts(images, videos, max_images: int, max_video_frames: int, prompt: str,
                         max_frame_side: int = DEFAULT_FRAME_SIDE, n_ctx: int = 0,
                         max_tokens: int = 0, text_tokens: int = 0, video_fps: int = 0) -> tuple:
    """OpenAI content-части из autogrow-пинов картинок и видео + статистика.
    Лимиты max_images/max_video_frames - НА КАЖДЫЙ пин своего типа; при известном контексте
    движка (n_ctx > 0) кадры заранее урезаются под окно (двигок сам не обрезает - отклоняет запрос).
    Возвращает (parts, stats): frames/images/videos (отправленные), est_tokens (~оценка
    vision-токенов), ctx, trimmed, note."""
    side = int(max_frame_side) if max_frame_side and int(max_frame_side) > 0 else DEFAULT_FRAME_SIDE
    groups = _plan_media(images, videos, int(max_images or 0), int(max_video_frames or 0))
    groups, note = _fit_media_to_context(groups, n_ctx, max_tokens, text_tokens, side)
    parts, total = _build_media_parts(groups, side, prompt, video_fps=int(video_fps or 0))
    stats = {
        "frames": total,
        "images": sum(len(g["idx"]) for g in groups if g["kind"] == "image"),
        "videos": sum(len(g["idx"]) for g in groups if g["kind"] == "video"),
        "est_tokens": _media_tokens(groups, side),
        "ctx": int(n_ctx or 0),
        "trimmed": note is not None,
        "note": note,
    }
    if total > MEDIA_WARN_FRAMES:
        print(f"[PolyStrata] ВНИМАНИЕ: {total} кадров за один запрос - контекст vision большой, "
              f"ответ может деградировать (ужмите max_images/max_video_frames)", flush=True)
    return parts, stats


# ================================================================ comfy: бронь VRAM
# Логика update_global_vram_reservation из llama.cpp-нод (LlamaCPP-Subprocess):
# фактическое потребление сервера (+ запас) пишется в comfy.model_management.
# EXTRA_RESERVED_VRAM (и в headroom DynamicVRAM/comfy-aimdo, если тот активен), поэтому
# ComfyUI планирует загрузку своих моделей, не выходя за пределы допустимого окна, и
# не ловит OOM. При выгрузке/сбросе исходные значения возвращаются.
_ORIG_EXTRA_RESERVED = {"bytes": None}     # исходное EXTRA_RESERVED_VRAM (до нашей брони)
_AIMDO_ACTIVE = {"on": False}              # headroom aimdo выставлен именно нами


def _aimdo_enabled() -> bool:
    """Активен ли DynamicVRAM (comfy-aimdo) - как в llama.cpp-нодах."""
    for mod in ("comfy.memory_management", "comfy.model_management"):
        try:
            m = __import__(mod, fromlist=["aimdo_enabled"])
            if getattr(m, "aimdo_enabled", False):
                return True
        except Exception:
            pass
    return False


def _sync_aimdo_headroom(reserved_bytes: int) -> bool:
    """Прокинуть резерв в DynamicVRAM (comfy-aimdo), если тот активен (best-effort)."""
    try:
        if not _aimdo_enabled():
            return False
        import comfy_aimdo.control as aimdo
        if getattr(aimdo, "lib", None) is None:
            return False
        aimdo.set_simple_vram_headroom(int(reserved_bytes))
        return True
    except Exception as exc:
        print(f"[PolyStrata] DynamicVRAM headroom не синхронизирован ({exc})", flush=True)
        return False


def update_global_vram_reservation(reserve_bytes: int) -> str:
    """Бронь VRAM для ComfyUI - метод llama.cpp-нод (update_global_vram_reservation):
    reserve_bytes > 0: comfy.model_management.EXTRA_RESERVED_VRAM = reserve_bytes
                       (+ headroom в comfy-aimdo, если активен);
    reserve_bytes = 0: вернуть исходные значения (используются при выгрузке модели).
    Возвращает человекочитаемую строку для лога; никогда не бросает исключений."""
    try:
        import comfy.model_management as mm
    except Exception as e:
        return f"EXTRA_RESERVED_VRAM не изменён (comfy.model_management недоступен: {e})"
    try:
        if _ORIG_EXTRA_RESERVED["bytes"] is None:
            _ORIG_EXTRA_RESERVED["bytes"] = int(getattr(mm, "EXTRA_RESERVED_VRAM", 0) or 0)
        if reserve_bytes and int(reserve_bytes) > 0:
            mm.EXTRA_RESERVED_VRAM = int(reserve_bytes)
            note = f"EXTRA_RESERVED_VRAM = {int(reserve_bytes) / 1048576:.0f} MiB"
            if _sync_aimdo_headroom(int(reserve_bytes)):
                _AIMDO_ACTIVE["on"] = True
                note += " + headroom DynamicVRAM (comfy-aimdo)"
            return note
        # --- снять бронь: вернуть исходные значения ComfyUI
        if _ORIG_EXTRA_RESERVED["bytes"] is not None:
            orig = _ORIG_EXTRA_RESERVED["bytes"]
            cur = int(getattr(mm, "EXTRA_RESERVED_VRAM", 0) or 0)
            if cur != orig:
                mm.EXTRA_RESERVED_VRAM = orig
                note = (f"EXTRA_RESERVED_VRAM возвращён ComfyUI: {cur / 1048576:.0f} -> "
                        f"{orig / 1048576:.0f} MiB")
            else:
                note = "EXTRA_RESERVED_VRAM уже исходный"
            if _AIMDO_ACTIVE["on"]:       # возвращаем только то, что выставили сами
                _sync_aimdo_headroom(orig)
                _AIMDO_ACTIVE["on"] = False
                note += " + headroom DynamicVRAM"
            _ORIG_EXTRA_RESERVED["bytes"] = None
            return note
        return "брони не было - ComfyUI не тронут"
    except Exception as e:
        return f"бронь VRAM ComfyUI не удалась ({e})"


# ================================================================ авто-выгрузка модели
# ТАК ЖЕ, как в llama.cpp-нодах (ComfyUI-LlamaCPP-Subprocess: хуки на
# PromptExecutor.execute и model_management.unload_all_models), только вместо kill
# процесса - POST /v1/unload. Зачем: главная нода забайпашена / воркфлоу переключён /
# ноды удалены - модель PolyStrata зря держит VRAM. Механика:
#   1) нода Chat при генерации помечает свой сервер как используемый (ACTIVE_SERVERS);
#   2) хук на PromptExecutor.execute (каждый запуск воркфлоу): серверы, которые есть в
#      ACTIVE_SERVERS, но НЕ упомянуты живыми нодами Chat/Unload текущего воркфлоу,
#      выгружаются автоматически (сервер-«сирота»);
#   3) хук на model_management.unload_all_models: ComfyUI освобождает память ВНЕ
#      выполнения нод (кнопка Free model and node cache, вытеснение) - выгружаем все
#      наши серверы. Внутри выполнения нод (execution.py в стеке) - НЕ трогаем.
# Сервер, к которому нодпак ни разу не обращался в этой сессии ComfyUI, НЕ выгружается
# никогда: к нему может быть подключён другой клиент (Jan, curl, другой ComfyUI).
# Выгрузка идёт в фоновом потоке: очередь ComfyUI не ждёт HTTP (wait до 30 c, если на
# сервере ещё идёт генерация - она ДОРАБАТАЕТ, затем модель выгрузится).
AUTO_UNLOAD_CLASSES = ("PolyStrataChat", "PolyStrataUnload")   # ноды с полем server
AUTO_UNLOAD_WAIT_S = 30.0        # сколько ждать активную генерацию перед выгрузкой
ACTIVE_SERVERS: set = set()      # нормализованные base-URL серверов этой сессии
_AUTO_LOCK = threading.Lock()
_AUTO_HOOKS = {"installed": False}


def mark_server_active(server: str) -> str:
    """Запомнить сервер как используемый нодами (вызывается из _chat_generate).
    Возвращает нормализованный base-URL ("http://ip:порт") или "" при ошибке."""
    try:
        base = PolyStrataClient(str(server or ""), "").base
    except Exception:
        return ""
    with _AUTO_LOCK:
        fresh = base not in ACTIVE_SERVERS
        ACTIVE_SERVERS.add(base)
    if fresh:
        print(f"[PolyStrata] авто-выгрузка: слежу за сервером {base} - выгружу его сам, "
              f"если он исчезнет из воркфлоу (байпасс главной ноды / другой воркфлоу)",
              flush=True)
    return base


def _auto_unload_server(base: str, wait_s: float = None) -> str:
    """POST /v1/unload; ошибки - в строку для лога, наверх не летят (фоновый механизм).
    Возвращает человекочитаемый итог: 'модель выгружена' / 'уже спит' / причина отказа."""
    if wait_s is None:
        wait_s = AUTO_UNLOAD_WAIT_S
    try:
        out = PolyStrataClient(base, "").unload(float(wait_s))
        st = str(out.get("status", "?"))
        if st == "unloaded":
            return "модель выгружена полностью"
        if st == "not loaded":
            return "модель уже спит"
        return f"неожиданный статус '{st}'"
    except PolyStrataError as e:
        return f"не удалось: {str(e)[:160]}"


def _servers_in_prompt(prompt) -> set:
    """Нормализованные адреса серверов из ЖИВЫХ нод Chat/Unload в промпте.
    Забайпашенные/замьюченные ноды НЕ считаются живыми: новый фронтенд ComfyUI шлёт
    их как class_type 'BypassedNode__<класс>' / 'MutedNode__<класс>' (точное совпадение
    с AUTO_UNLOAD_CLASSES не пройдёт), старый фронтенд вообще не включает их в промпт."""
    found = set()
    if not isinstance(prompt, dict):
        return found
    for info in prompt.values():
        if not isinstance(info, dict) or info.get("mode") in (2, 4):   # mute/bypass (старый фронт)
            continue
        if info.get("class_type") in AUTO_UNLOAD_CLASSES:
            srv = (info.get("inputs") or {}).get("server", "")
            if srv:
                try:
                    found.add(PolyStrataClient(str(srv), "").base)
                except Exception:
                    pass
    return found


def _auto_unload_for_prompt(prompt) -> list:
    """Выгрузить серверы-«сироты»: использовались нодами раньше, но их НЕТ среди живых
    нод текущего воркфлоу. Сервер, выгрузить который не удалось (занят), остаётся под
    наблюдением - попытка повторится при следующем запуске воркфлоу. Возвращает лог."""
    with _AUTO_LOCK:
        tracked = set(ACTIVE_SERVERS)
    if not tracked:
        return []
    orphans = sorted(tracked - _servers_in_prompt(prompt))
    msgs = []
    for base in orphans:
        note = _auto_unload_server(base)
        done = note.startswith(("модель выгружена", "модель уже спит"))
        if done:
            with _AUTO_LOCK:
                ACTIVE_SERVERS.discard(base)
        msgs.append(f"[PolyStrata] АВТО-ВЫГРУЗКА: сервер {base} отсутствует в текущем воркфлоу "
                    f"(главная нода забайпашена / другой воркфлоу / ноды удалены) - {note}"
                    + ("" if done else "; остаётся под наблюдением"))
    if any("модель выгружена" in m for m in msgs):
        msgs.append(f"[PolyStrata] АВТО-ВЫГРУЗКА: ComfyUI: {update_global_vram_reservation(0)}.")
    for m in msgs:
        print(m, flush=True)
    return msgs


def _auto_unload_all() -> list:
    """Выгрузить ВСЕ серверы, использованные нодами в этой сессии (хук
    unload_all_models: ComfyUI сам освобождает память вне выполнения нод)."""
    with _AUTO_LOCK:
        tracked = set(ACTIVE_SERVERS)
    msgs = []
    for base in sorted(tracked):
        note = _auto_unload_server(base)
        if note.startswith(("модель выгружена", "модель уже спит")):
            with _AUTO_LOCK:
                ACTIVE_SERVERS.discard(base)
        msgs.append(f"[PolyStrata] АВТО-ВЫГРУЗКА (ComfyUI освободил память): {base} - {note}")
    if msgs and any("модель выгружена" in m for m in msgs):
        msgs.append(f"[PolyStrata] АВТО-ВЫГРУЗКА: ComfyUI: {update_global_vram_reservation(0)}.")
    for m in msgs:
        print(m, flush=True)
    return msgs


def install_auto_unload_hooks() -> bool:
    """Установить хуки - механика llama.cpp-нод (hooked_execute /
    hooked_unload_all_models_llamacpp). Повторные вызовы безопасны; без ComfyUI False."""
    if _AUTO_HOOKS["installed"]:
        return True
    try:
        import inspect
        import execution
        import comfy.model_management as mm
    except Exception as e:
        print(f"[PolyStrata] авто-выгрузка: хуки не установлены (ComfyUI недоступен: {e})",
              flush=True)
        return False

    if not hasattr(execution.PromptExecutor, "_orig_execute_polystrata"):
        execution.PromptExecutor._orig_execute_polystrata = execution.PromptExecutor.execute

        def _hooked_execute(self, prompt, prompt_id, *args, **kwargs):
            try:
                note_workflow_start()   # новый запуск: тоггл clear_comfy_vram снова доступен
                threading.Thread(target=_auto_unload_for_prompt, args=(prompt,),
                                 daemon=True).start()   # очередь не ждёт HTTP
            except Exception as e:
                print(f"[PolyStrata] авто-выгрузка: ошибка хука execute ({e})", flush=True)
            return self._orig_execute_polystrata(prompt, prompt_id, *args, **kwargs)

        execution.PromptExecutor.execute = _hooked_execute

    if not hasattr(mm, "_orig_unload_all_polystrata"):
        mm._orig_unload_all_polystrata = mm.unload_all_models

        def _hooked_unload_all(*args, **kwargs):
            try:
                stack = inspect.stack()
                # не зацикливаться: во время выполнения нод (execution.py в стеке) не трогаем
                is_node_execution = any("execution.py" in fr.filename for fr in stack)
            except Exception:
                is_node_execution = False
            if not is_node_execution:
                _auto_unload_all()
            return mm._orig_unload_all_polystrata(*args, **kwargs)

        mm.unload_all_models = _hooked_unload_all

    _AUTO_HOOKS["installed"] = True
    print("[PolyStrata] авто-выгрузка активна: главная нода забайпашена / открыт другой "
          "воркфлоу - модель выгружается сама (как в llama.cpp-нодах)", flush=True)
    return True


# ================================================================ очистка памяти ComfyUI (тоггл)
# Виджет clear_comfy_vram ноды Chat: перед генерацией выгрузить модели САМОГО ComfyUI
# (checkpoint/LoRA/ControlNet - всё, что ComfyUI держит в VRAM) и почистить torch-кэш -
# механика кнопки "Free model and node cache" (mm.unload_all_models) + soft_empty_cache + gc.
# Модель PolyStrata это НЕ задевает: наш хук unload_all_models внутри выполнения нод
# (execution.py в стеке) серверы PolyStrata не трогает.
#
# «НЕ ОЧИЩАТЬ МНОГО РАЗ»: листы на входах (циклы list-паков, батчи картинок, перебор сидов,
# несколько нод Chat) исполняют ноду многократно в ОДНОМ запуске воркфлоу - очистка делается
# только на ПЕРВОМ исполнении (_COMFY_CLEAR), флаг сбрасывается в начале КАЖДОГО запуска
# воркфлоу (note_workflow_start из хука PromptExecutor.execute - того же, что у авто-выгрузки).
_COMFY_CLEAR = {"done": False}     # очистка уже сделана в текущем запуске воркфлоу


def note_workflow_start() -> None:
    """Начался новый запуск воркфлоу: флаг «память ComfyUI уже очищена» сброшен.
    Вызывается хуком PromptExecutor.execute (см. install_auto_unload_hooks); безопасно
    звать и напрямую (повторные вызовы - no-op)."""
    with _AUTO_LOCK:
        _COMFY_CLEAR["done"] = False


def free_comfy_model_memory() -> str:
    """Выгрузить модели ComfyUI + почистить torch-кэш (как кнопка 'Free model and node cache').
    Модели PolyStrata не задеваются (см. секцию авто-выгрузки). Возвращает человекочитаемую
    строку для лога; исключений не бросает - очистка не должна ронять генерацию."""
    try:
        import comfy.model_management as mm
    except Exception as e:
        return f"не удалось: comfy.model_management недоступен ({e})"
    notes = []
    unload = getattr(mm, "unload_all_models", None)
    if unload is not None:
        try:
            unload()
            notes.append("модели ComfyUI выгружены (unload_all_models)")
        except Exception as e:
            notes.append(f"выгрузка моделей не удалась: {e}")
    else:
        notes.append("unload_all_models недоступен в этой версии ComfyUI")
    try:
        cache_fn = getattr(mm, "soft_empty_cache", None) or getattr(mm, "empty_cache", None)
        if cache_fn is not None:
            cache_fn()
            notes.append("torch-кэш очищен")
    except Exception as e:
        notes.append(f"torch-кэш не очищен: {e}")
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    return "; ".join(notes)


def _maybe_clear_comfy_vram(enabled) -> str:
    """Тоггл clear_comfy_vram: очистить память моделей ComfyUI ОДИН РАЗ на запуск воркфлоу.
    Лист на входе (цикл по картинкам/сидам, несколько нод Chat) исполняет ноду многократно -
    повторные очистки бессмысленны: после первой выгружать нечего, результат тот же. Флаг
    сбрасывается в начале каждого запуска воркфлоу (note_workflow_start). Если хуки не
    установились (comfy недоступен - тесты), сбрасывать флаг некому - тогда очистка делается
    на каждом вызове (безопасный запасной путь, никогда не 'залипает')."""
    if not enabled:
        return ""
    with _AUTO_LOCK:
        done = _COMFY_CLEAR["done"]
        _COMFY_CLEAR["done"] = True
    if done and _AUTO_HOOKS["installed"]:
        print("[PolyStrata] очистка памяти ComfyUI: уже сделана в этом запуске воркфлоу "
              "(лист/цикл/повторное исполнение) - пропуск", flush=True)
        return "уже очищено в этом запуске"
    note = free_comfy_model_memory()
    print(f"[PolyStrata] очистка памяти ComfyUI перед генерацией: {note}", flush=True)
    return note


# ================================================================ общее ядро Chat
def _parse_logit_bias(text: str) -> dict:
    """Строки 'id' (бан, -100) / 'id:bias' / 'id=bias' / 'id bias' -> {'id': float}."""
    out: dict = {}
    for ln in str(text or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"^(\d+)\s*(?::|=|\s)\s*(-?\d+(?:\.\d+)?)$", s)
        if m:
            tid, bias = m.group(1), float(m.group(2))
            if not -100.0 <= bias <= 100.0:
                raise PolyStrataError(f"logit_bias: bias {bias} вне диапазона -100..100 (строка '{s}')")
        elif re.fullmatch(r"\d+", s):
            tid, bias = s, -100.0
        else:
            raise PolyStrataError(f"logit_bias: не понял строку '{s}' - формат: '<token_id>' "
                                  f"(запрет) или '<token_id>:<-100..100>'")
        if tid in out and out[tid] != bias:
            raise PolyStrataError(f"logit_bias: токен {tid} указан дважды с разными bias")
        out[tid] = float(bias)
    return out


def _parse_extra_sampling(text: str) -> dict:
    """JSON-объект extra_sampling -> dict (сливается в тело запроса)."""
    s = str(text or "").strip()
    if not s:
        return {}
    try:
        val = json.loads(s)
    except ValueError as e:
        raise PolyStrataError(f"extra_sampling: неверный JSON ({e})") from e
    if not isinstance(val, dict):
        raise PolyStrataError("extra_sampling: ожидаю JSON-ОБЪЕКТ {\"поле\": значение}")
    return val


_SERVER_CTX_CACHE = {}     # base сервера -> (started движка, n_ctx)


def _server_ctx(client: PolyStrataClient) -> int:
    """Окно контекста движка в токенах (для префлайта кадров): /v1/status (context.native или
    cache_max_tokens), запасной путь - /props (default_generation_settings.n_ctx).
    Кэш по адресу сервера и времени старта движка (перезапуск с другим --max-context обновит).
    0 = узнать не удалось - префлайт отключается, остаются только лимиты кадров на пин."""
    try:
        st = client.status()
        started = st.get("started")
        cached = _SERVER_CTX_CACHE.get(client.base)
        if cached and cached[0] == started:
            return int(cached[1])
        n = 0
        for v in ((st.get("context") or {}).get("native"), st.get("cache_max_tokens")):
            try:
                n = int(v or 0)
            except (TypeError, ValueError):
                n = 0
            if n > 0:
                break
        if n <= 0:
            n = int(((client.props().get("default_generation_settings") or {}).get("n_ctx")) or 0)
        if n > 0:
            _SERVER_CTX_CACHE[client.base] = (started, n)
        return n
    except Exception:
        return 0


def _chat_generate(server: str = "127.0.0.1:8080", api_key: str = "", prompt: str = "",
                   system_prompt: str = "", model: str = "", reasoning: str = "default",
                   max_tokens: int = 1024, temperature: float = 0.8, top_p: float = 0.95,
                   seed: int = 0, reserve_mib: int = 0, extra_mib: int = 0, images=None,
                   videos=None, max_images: int = 16, max_video_frames: int = 8,
                   max_frame_side: int = DEFAULT_FRAME_SIDE, memory=None, sampling=None,
                   video_fps: int = 0, clear_comfy_vram: bool = False) -> tuple:
    """Единое ядро генерации для V3-нода (execute) и V1-фолбэка (generate).
    Возвращает (text, thoughts). Кнопка Cancel (interrupt) ComfyUI прерывает на любом шаге:
    во время запроса сокет закрывается ПО-НАСТОЯЩЕМУ (FIN), и watch-поток PolyStrata
    (_watch_client: select + MSG_PEEK каждые 0.5 c) отменяет генерацию движка.
    Входы приходят УЖЕ нормализованными: на входе ноды _scalars прогоняет весь kwargs
    (имена виджетов нигде не хардкодятся - любые списки от list-паков заменены первыми
    элементами, медиа-списки разворачивает в пины _as_pins); тоггл clear_comfy_vram
    выгружает модели ComfyUI ОДИН РАЗ на запуск воркфлоу."""
    _throw_if_interrupted()                      # прервано ДО старта - сервер не дёргаем
    mark_server_active(server)                   # авто-выгрузка: сервер под наблюдением
    # --- тоггл: очистить память моделей ComfyUI перед генерацией (один раз на запуск воркфлоу)
    _maybe_clear_comfy_vram(clear_comfy_vram)
    _throw_if_interrupted()
    store = memory if hasattr(memory, "remember") else None
    sd = dict(sampling) if isinstance(sampling, dict) else {}

    # --- reserve VRAM для ComfyUI (ДО генерации; 0 = не трогаем сервер вовсе)
    reserve_note = ""
    if reserve_mib and int(reserve_mib) > 0:
        client0 = PolyStrataClient(server, api_key)
        reserve_note = _apply_reserve(client0, int(reserve_mib), int(extra_mib or 0),
                                      stop_check=_throw_if_interrupted)
        _throw_if_interrupted()                  # выгрузка сегментов кэша может быть долгой

    # --- история/системная строка (нужны ДО оценки объёма текста для префлайта контекста)
    sys_line = store.system_line() if store is not None else ""
    if not sys_line:
        sys_line = (system_prompt or "").strip()
    hist = store.history() if store is not None else []

    # --- контекст движка (best-effort) -> префлайт: кадры подгоняются под окно заранее
    client = PolyStrataClient(server, api_key)
    n_ctx = _server_ctx(client)
    text_chars = len(sys_line) + len(prompt or "") + \
        sum(len(str(m.get("content") or "")) for m in hist)
    text_tokens = int(text_chars / TEXT_CHARS_PER_TOKEN) + 1

    # --- медиа: картинки и видео отдельно (autogrow-пины), лимиты НА ПИН, подгонка под окно
    parts, stats = _collect_media_parts(images, videos, int(max_images or 0),
                                        int(max_video_frames or 0), prompt,
                                        max_frame_side=int(max_frame_side or 0), n_ctx=n_ctx,
                                        max_tokens=int(max_tokens or 0), text_tokens=text_tokens,
                                        video_fps=int(video_fps or 0))
    _throw_if_interrupted()                      # кодирование кадров может быть долгим

    messages = []
    if sys_line:
        messages.append({"role": "system", "content": sys_line})
    messages.extend(hist)
    has_media = len(parts) > 1
    messages.append({"role": "user", "content": parts if has_media else prompt})

    # сэмплинг: виджеты этой ноды - база; подключённая нода Sampling перекрывает
    kw = {"temperature": temperature, "top_p": top_p, "reasoning": reasoning, "seed": seed}
    kw.update({k: v for k, v in sd.items() if v is not None})

    notes = []
    if store is not None:
        notes.append(f"память: {store.turns()} ход(ов)")
    if sampling:
        notes.append("сэмплинг: из суб-ноды")
    if reserve_note:
        notes.append("reserve VRAM применён")
    note = ("; " + ", ".join(notes)) if notes else ""
    ctx_line = f", контекст: {n_ctx}" if n_ctx else ""
    est_line = f", ~{stats['est_tokens']} ток. медиа" if stats["frames"] else ""
    print(f"[PolyStrata] chat -> {client.base.removeprefix('http://')} "
          f"(кадров: {stats['frames']} [изобр.: {stats['images']}, видео: {stats['videos']}]"
          f"{est_line}{ctx_line}, max_tokens: {max_tokens or '∞'}{note})", flush=True)
    if stats.get("note"):
        print(f"[PolyStrata] {stats['note']}", flush=True)
    if stats["frames"] and not n_ctx:
        print("[PolyStrata] контекст движка неизвестен (сервер не ответил?) - префлайт окна "
              "пропущен, работают только лимиты кадров на пин", flush=True)
    try:
        resp = client.chat(messages, model=model, max_tokens=max_tokens, stop_check=_throw_if_interrupted, **kw)
    except _Interrupted:
        print("[PolyStrata] ПРЕРВАНО (Cancel ComfyUI): соединение закрыто - сервер отменяет "
              "генерацию; ход в память не записан", flush=True)
        raise
    try:
        choice = resp["choices"][0]
        msg = choice.get("message", {}) or {}
        text = msg.get("content") or ""
        thoughts = msg.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise PolyStrataError(f"Неожиданный ответ сервера (нет choices[0].message.content): "
                              f"{str(resp)[:400]}") from e
    if store is not None:
        kept = store.remember(prompt, text)     # в историю - только текст (без медиа)
        print(f"[PolyStrata] память: сохранён ход, в истории {kept} ход(ов)", flush=True)
    usage = resp.get("usage") or {}
    ctok = usage.get("completion_tokens")
    extra = f", токены: {ctok}" if ctok else ""
    th = f", размышления: {len(thoughts)} символов" if thoughts else ""
    print(f"[PolyStrata] ответ: {len(text)} символов{th}{extra}", flush=True)
    if reserve_note:
        print(f"[PolyStrata] reserve VRAM: {reserve_note}", flush=True)
    return text, thoughts


def _apply_reserve(client: PolyStrataClient, reserve_mib: int, extra_mib: int,
                   stop_check=None) -> str:
    """POST /v1/vram: держать reserve+extra MiB VRAM свободными для ComfyUI, затем -
    бронь ComfyUI по методу llama.cpp-нод: ФАКТИЧЕСКОЕ потребление PolyStrata
    (total GPU - фактически свободно) + extra_mib -> EXTRA_RESERVED_VRAM (+ aimdo).
    Ошибки НЕ роняют генерацию (резерв - пожелание к скорости, а не условие);
    stop_check позволяет прервать долгую выгрузку сегментов кэша кнопкой Cancel."""
    target = reserve_mib + max(0, extra_mib)
    lines = []
    total_mib = None
    try:
        st = client.status()
        total_mib = ((st.get("machine") or {}).get("gpu") or {}).get("total_mib")
        v = st.get("vram") or {}
        cache, full = v.get("expert_cache_mib"), v.get("expert_cache_full_mib")
        if cache is not None:
            lines.append(f"модель в VRAM (кэш экспертов): {cache} / {full} MiB")
    except PolyStrataError as e:
        lines.append(f"состояние сервера узнать не удалось ({e})")
    try:
        out = client.vram(target, stop_check=stop_check)
    except PolyStrataError as e:
        lines.append(f"РЕЗЕРВ НЕ ПРИМЕНЁН: {e} - генерация продолжится без него")
        return "; ".join(lines)
    if out.get("status") == "not loaded":
        # модель спит: на GPU ничего нет - бронировать для ComfyUI нечего
        lines.append(f"модель спит: свободными держать {target} MiB записано и применится "
                     f"при следующей загрузке (reserve {reserve_mib}"
                     f"{f' + extra {extra_mib}' if extra_mib else ''})")
        lines.append(f"ComfyUI: {update_global_vram_reservation(0)}")
        return "; ".join(lines)
    free = out.get("vram_free_mib")
    cache, full = out.get("expert_cache_mib"), out.get("expert_cache_full_mib")
    slots = out.get("expert_slots")
    lines.append(f"цель свободной VRAM: {target} MiB (reserve {reserve_mib}"
                 f"{f' + extra {extra_mib}' if extra_mib else ''}), фактически свободно {free} MiB")
    if free is not None and free < target:
        lines.append("ЧАСТИЧНОЕ ОСВОБОЖДЕНИЕ: свободно меньше цели - кэш экспертов исчерпан "
                     "(для полной отдачи есть нода 'PolyStrata Unload')")
    # --- бронь ComfyUI: АДЕКВАТНОЕ (фактическое) значение, а не запрошенная цель
    if total_mib and free is not None:
        held_mib = max(0, int(total_mib) - int(free))
        note = update_global_vram_reservation((held_mib + max(0, extra_mib)) << 20)
        lines.append(f"PolyStrata держит {held_mib} MiB; ComfyUI: {note}")
    else:
        lines.append("ComfyUI: total VRAM сервера неизвестен - EXTRA_RESERVED_VRAM не изменён "
                     "(обновите сервер: GET /v1/status без machine.gpu)")
    if cache is not None:
        lines.append(f"кэш экспертов в VRAM: {cache} / {full} MiB"
                     f"{f' (слотов: {slots})' if slots is not None else ''}")
    return "; ".join(lines)


# ================================================================ нода: Chat (V3)
def _seed_input(tooltip: str):
    """Int-вход seed; control_after_generate добавляется, если comfy_api его поддерживает."""
    kw = dict(default=0, min=0, max=4294967295, step=1, tooltip=tooltip)
    try:
        return comfy_io.Int.Input("seed", control_after_generate=True, **kw)
    except TypeError:                        # старый comfy_api без control_after_generate
        return comfy_io.Int.Input("seed", **kw)


if _HAS_V3:

    class PolyStrataChat(comfy_io.ComfyNode):
        """Генерация текста через сервер PolyStrata. SCHEMA V3 (comfy_api.latest):
        динамические (autogrow) входы картинок img_* и ОТДЕЛЬНО видео vid_* - как в
        llama.cpp-нодах. Без стриминга: ответ целиком. Одиночный запрос сам по себе;
        история - опциональная нода Memory, тонкий сэмплинг - нода Sampling.

        reasoning/temperature/top_p/seed живут ТОЛЬКО здесь (в Sampling их нет).
        Спящая модель (после Unload) просыпается сама: сервер грузит движок по требованию.
        reserve_mib > 0 - перед генерацией отдать VRAM ComfyUI (пониженное потребление)
        и выставить бронь EXTRA_RESERVED_VRAM по факту; 0 - не влиять.
        clear_comfy_vram - тоггл: перед генерацией выгрузить модели САМОГО ComfyUI
        (checkpoint/LoRA и пр.) и почистить torch-кэш; в одном запуске воркфлоу - ОДИН РАЗ
        (лист на входе / цикл / несколько нод Chat повторно память не дёргают).
        max_images/max_video_frames - лимиты НА КАЖДЫЙ пин своего типа; кадры заранее
        подгоняются под контекст движка (узнаётся из /v1/status), потому что двигок запросы
        не обрезает. Кнопка Cancel (interrupt) ComfyUI прерывает генерацию: соединение
        закрывается, сервер отменяет запрос.
        """

        @classmethod
        def define_schema(cls):
            image_pin = comfy_io.Image.Input("img", optional=True, tooltip=IMAGES_TOOLTIP)
            image_template = comfy_io.Autogrow.TemplatePrefix(image_pin, prefix="img",
                                                              min=0, max=MAX_PIN_SLOTS)
            video_pin = comfy_io.Image.Input("vid", optional=True, tooltip=VIDEOS_TOOLTIP)
            video_template = comfy_io.Autogrow.TemplatePrefix(video_pin, prefix="vid",
                                                              min=0, max=MAX_PIN_SLOTS)
            return comfy_io.Schema(
                node_id="PolyStrataChat",
                display_name="PolyStrata Chat",
                category="PolyStrata",
                description=("Чат с сервером PolyStrata (без стриминга). Schema V3: autogrow-входы "
                             "картинок (img_*) и видео (vid_*) отдельно, каждый vid-пин = отдельное "
                             "видео кадрами (как Jan). Память и тонкий сэмплинг - суб-нодами."),
                inputs=[
                    comfy_io.String.Input("server", default="127.0.0.1:8080", tooltip=SERVER_TOOLTIP),
                    comfy_io.String.Input("prompt", multiline=True,
                                          default="Опиши это видео или изображения подробно.",
                                          tooltip="Запрос к модели (роль user)."),
                    comfy_io.Combo.Input("system_prompt_preset", options=system_prompt_options(),
                                         default=NO_SYSTEM_PROMPT,
                                         tooltip="Пресет системного промпта из ComfyUI/models/LLM/prompts "
                                                 "(.txt, можно в подпапках) - тот же путь, что в llama.cpp-нодах. "
                                                 "none - без пресета. Выбранный пресет ПЕРЕКРЫВАЕТ поле "
                                                 "system_prompt_text (но перекрывается системным промптом ноды "
                                                 "Memory, если тот задан). Список перечитывается при каждой "
                                                 "перезагрузке страницы ComfyUI - свежие .txt файлы видны сразу."),
                    comfy_io.String.Input("system_prompt_text", multiline=True, default="",
                                          tooltip="Ручной системный промпт (роль system). Используется, если "
                                                  "пресет = none. Пусто - не отправляется. Перекрывается "
                                                  "системным промптом ноды Memory, если тот задан."),
                    comfy_io.String.Input("model", default="",
                                          tooltip="Имя модели для поля 'model'. Пусто - не отправляется "
                                                  "(сервер использует запущенную). Имя должно совпадать с "
                                                  "запущенной, иначе сервер вернёт 404."),
                    comfy_io.Combo.Input("reasoning",
                                         options=["default", "none", "low", "medium", "high"],
                                         default="default",
                                         tooltip="Размышления думающих моделей (Qwen/GLM/Gemma...). "
                                                 "default - как на сервере; none - ВЫКЛЮЧИТЬ размышления "
                                                 "(шлётся reasoning_effort: none + chat_template_kwargs "
                                                 "enable_thinking=false - оба механизма, которые сервер "
                                                 "реально читает); low/medium/high - reasoning_effort "
                                                 "(high сервер поднимает до xhigh). Размышления приходят "
                                                 "отдельным выходом 'thoughts'."),
                    comfy_io.Int.Input("max_tokens", default=1024, min=0, max=131072, step=1,
                                       tooltip="Лимит токенов ответа. 0 = без ограничения (сколько "
                                               "влезет в контекст после промпта)."),
                    comfy_io.Float.Input("temperature", default=0.8, min=0.0, max=2.0, step=0.05,
                                         tooltip="Температура сэмплинга (0 = жадный режим). Игнорируется, "
                                                 "если подключён Sampling и там задана своя."),
                    comfy_io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01,
                                         tooltip="Top-P (nucleus). Игнорируется, если подключён Sampling "
                                                 "и там задан свой."),
                    _seed_input(SEED_TOOLTIP),
                    comfy_io.Int.Input("max_frame_side", default=DEFAULT_FRAME_SIDE, min=0, max=2048,
                                       step=64, tooltip=FRAME_SIDE_TOOLTIP),
                    comfy_io.Int.Input("reserve_mib", default=0, min=0, max=524288, step=512,
                                       tooltip=RESERVE_TOOLTIP),
                    comfy_io.Int.Input("extra_mib", default=0, min=0, max=262144, step=64,
                                       tooltip=EXTRA_TOOLTIP),
                    comfy_io.Boolean.Input("clear_comfy_vram", default=False,
                                           tooltip=CLEAR_VRAM_TOOLTIP),
                    comfy_io.String.Input("api_key", default="", tooltip=API_KEY_TOOLTIP),
                    comfy_io.Int.Input("max_images", default=16, min=0, max=MAX_IMAGES_PER_PIN,
                                       step=1, tooltip=MAX_IMAGES_TOOLTIP),
                    comfy_io.Int.Input("max_video_frames", default=8, min=0,
                                       max=MAX_VIDEO_FRAMES_PER_PIN, step=1,
                                       tooltip=MAX_VIDEO_FRAMES_TOOLTIP),
                    comfy_io.Int.Input("video_fps", default=0, min=0, max=240, step=1,
                                       tooltip=VIDEO_FPS_TOOLTIP),
                    comfy_io.Autogrow.Input("images", template=image_template),
                    comfy_io.Autogrow.Input("videos", template=video_template),
                    comfy_io.Custom(PS_MEMORY).Input("memory", optional=True,
                                                     tooltip="Опционально: нода 'PolyStrata Memory' - история "
                                                             "разговора (multi-turn). Без неё - одиночный запрос. "
                                                             "В историю пишется только текст (медиа не попадают)."),
                    comfy_io.Custom(PS_SAMPLING).Input("sampling", optional=True,
                                                       tooltip="Опционально: нода 'PolyStrata Sampling' - top_k, "
                                                               "min_p, штрафы, stop, seed, logit_bias, extra_sampling "
                                                               "(temperature/top_p/reasoning - виджеты этой ноды). "
                                                               "Заданное в Sampling перекрывает виджеты чата."),
                ],
                outputs=[
                    comfy_io.String.Output(display_name="text"),
                    comfy_io.String.Output(display_name="thoughts"),
                ],
            )

        @classmethod
        def execute(cls, **kw) -> comfy_io.NodeOutput:
            """Входы приходят с именами виджетов из схемы (в т.ч. списками от list-паков).
            Имена НЕ перечисляются: _scalars нормализует весь kwargs разом, дальше ядро
            получает готовые скаляры. Новый виджет в схеме защищён от листов автоматически."""
            kw = _scalars(kw)
            kw["system_prompt"] = resolve_system_prompt(
                kw.pop("system_prompt_preset", NO_SYSTEM_PROMPT),
                kw.pop("system_prompt_text", ""))
            text, thoughts = _chat_generate(**kw)
            return comfy_io.NodeOutput(text, thoughts)

else:

    class PolyStrataChat:                    # V1-фолбэк: старый ComfyUI без comfy_api
        """V1-версия ноды Chat (ComfyUI без schema V3): вместо autogrow-пинов - два
        фиксированных входа images/videos (IMAGE, можно батч). Поведение то же."""

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "server": ("STRING", {"default": "127.0.0.1:8080", "tooltip": SERVER_TOOLTIP}),
                    "prompt": ("STRING", {"multiline": True,
                                          "default": "Опиши это видео или изображения подробно.",
                                          "tooltip": "Запрос к модели (роль user)."}),
                    "system_prompt_preset": (system_prompt_options(),
                                             {"default": NO_SYSTEM_PROMPT,
                                              "tooltip": "Пресет системного промпта из models/LLM/prompts (.txt) - "
                                                         "тот же путь, что в llama.cpp-нодах. Выбранный пресет "
                                                         "ПЕРЕКРЫВАЕТ system_prompt_text. Список обновляется при "
                                                         "перезагрузке страницы ComfyUI."}),
                    "system_prompt_text": ("STRING", {"multiline": True, "default": "",
                                                      "tooltip": "Ручной системный промпт (роль system). "
                                                                 "Используется, если пресет = none."}),
                    "model": ("STRING", {"default": "", "tooltip": "Имя модели; пусто = запущенная."}),
                    "reasoning": (["default", "none", "low", "medium", "high"],
                                  {"default": "default",
                                   "tooltip": "Размышления думающих моделей (Qwen/GLM/Gemma...).\n"
                                              "default - как на сервере; none - ВЫКЛЮЧИТЬ размышления "
                                              "(reasoning_effort: none + chat_template_kwargs "
                                              "enable_thinking=false); low/medium/high - reasoning_effort "
                                              "(high -> xhigh)."}),
                    "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 131072, "step": 1,
                                           "tooltip": "Лимит токенов ответа. 0 = без ограничения."}),
                    "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05,
                                              "tooltip": "Температура (0 = жадный)."}),
                    "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Top-P (nucleus)."}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295, "step": 1,
                                     "control_after_generate": True, "tooltip": SEED_TOOLTIP}),
                    "max_frame_side": ("INT", {"default": DEFAULT_FRAME_SIDE, "min": 0, "max": 2048,
                                               "step": 64, "tooltip": FRAME_SIDE_TOOLTIP}),
                    "reserve_mib": ("INT", {"default": 0, "min": 0, "max": 524288, "step": 512,
                                            "tooltip": RESERVE_TOOLTIP}),
                    "extra_mib": ("INT", {"default": 0, "min": 0, "max": 262144, "step": 64,
                                          "tooltip": EXTRA_TOOLTIP}),
                    "clear_comfy_vram": ("BOOLEAN", {"default": False, "tooltip": CLEAR_VRAM_TOOLTIP}),
                    "api_key": ("STRING", {"default": "", "tooltip": API_KEY_TOOLTIP}),
                    "max_images": ("INT", {"default": 16, "min": 0, "max": MAX_IMAGES_PER_PIN,
                                           "step": 1, "tooltip": MAX_IMAGES_TOOLTIP}),
                    "max_video_frames": ("INT", {"default": 8, "min": 0,
                                                 "max": MAX_VIDEO_FRAMES_PER_PIN, "step": 1,
                                                 "tooltip": MAX_VIDEO_FRAMES_TOOLTIP}),
                    "video_fps": ("INT", {"default": 0, "min": 0, "max": 240, "step": 1,
                                         "tooltip": VIDEO_FPS_TOOLTIP}),
                },
                "optional": {
                    "images": ("IMAGE", {"tooltip": "Картинки или кадры видео (IMAGE, можно батч). "
                                                    "Автогроу-пины доступны в schema V3 (ComfyUI с comfy_api)."}),
                    "videos": ("IMAGE", {"tooltip": "Видео батчем кадров (IMAGE [N,H,W,C]) - уходит "
                                                    "отдельным блоком '[Video 1]'; при заданном video_fps - "
                                                    "с метками времени '<X.X seconds>' между кадрами."}),
                    "memory": (PS_MEMORY, {"tooltip": "Опционально: нода 'PolyStrata Memory'."}),
                    "sampling": (PS_SAMPLING, {"tooltip": "Опционально: нода 'PolyStrata Sampling'."}),
                },
            }

        RETURN_TYPES = ("STRING", "STRING")
        RETURN_NAMES = ("text", "thoughts")
        FUNCTION = "generate"
        CATEGORY = "PolyStrata"

        def generate(self, **kw):
            """Входы приходят с именами виджетов из INPUT_TYPES (в т.ч. списками от list-паков).
            Имена НЕ перечисляются: _scalars нормализует весь kwargs разом, дальше ядро
            получает готовые скаляры. Новый виджет в INPUT_TYPES защищён от листов автоматически."""
            kw = _scalars(kw)
            kw["system_prompt"] = resolve_system_prompt(
                kw.pop("system_prompt_preset", NO_SYSTEM_PROMPT),
                kw.pop("system_prompt_text", ""))
            text, thoughts = _chat_generate(**kw)
            return (text, thoughts)


# ================================================================ нода: Sampling
class PolyStrataSampling:
    """Тонкий сэмплинг для ноды Chat - ТОЛЬКО то, чего нет в виджетах чата
    (temperature/top_p/reasoning/seed живут в ноде Chat, здесь их нет - без дубликатов).

    Подключается выходом 'sampling' ко входу 'sampling' ноды Chat; заданные значения
    перекрывают виджеты чата, незаданные (-1/пусто) - пропускаются.

    Поля /v1/chat/completions PolyStrata (serve/server.py, check_request_sampling +
    StrataEngine.sampling_keys): top_k (1..64), min_p, repetition_penalty (repeat_penalty),
    penalty_last_n (repeat_last_n), presence_penalty, frequency_penalty, stop (до 4 строк),
    logit_bias (запрет токенов - как banned_tokens в llama-нодах).
    Семплеров xtc/dry/dynatemp/mirostat/tfs_z/typical_p/smoothing/top_n_sigma у движка НЕТ -
    сервер молча отбрасывает их; на будущее - extra_sampling (JSON без обновления нод)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "top_k": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1,
                                  "tooltip": "-1 = не задано (действует чат/сервер); 1..64 "
                                             "(движок ограничивает сверху 64)."}),
                "min_p": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "-1 = не задано; 0..1. Отсекает токены с вероятностью "
                                               "меньше min_p от максимума (альтернатива top_p)."}),
                "repeat_penalty": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.01,
                                             "tooltip": "Штраф повторов (repetition_penalty). -1 = не задано; "
                                                        "1 = выключен; >1 - реже повторять (обычно 1.05..1.3)."}),
                "repeat_last_n": ("INT", {"default": -1, "min": -1, "max": 2048, "step": 1,
                                          "tooltip": "Окно штрафов повторов (penalty_last_n), токенов. "
                                                     "-1 = не задано; 1..2048 - размер окна."}),
                "presence_penalty": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.01,
                                               "tooltip": "Штраф за уже встречавшиеся токены. -1 = не задано; "
                                                          "0 = выключен (сервер принимает >= 0)."}),
                "frequency_penalty": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.01,
                                                "tooltip": "Частотный штраф. -1 = не задано; 0 = выключен "
                                                           "(сервер принимает >= 0)."}),
                "stop": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "Стоп-строки, по строке на каждую (сервер берёт до 4). "
                                               "Пусто - не задано. Генерация остановится перед совпадением."}),
                "logit_bias": ("STRING", {"multiline": True, "default": "",
                                          "tooltip": LOGIT_BIAS_TOOLTIP}),
                "extra_sampling": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": EXTRA_SAMPLING_TOOLTIP}),
            },
        }

    RETURN_TYPES = (PS_SAMPLING,)
    RETURN_NAMES = ("sampling",)
    FUNCTION = "run"
    CATEGORY = "PolyStrata"
    DESCRIPTION = ("Тонкий сэмплинг для ноды Chat (без дубликатов: temperature/top_p/reasoning/seed - "
                   "в чате). -1/пусто = поле не задано. logit_bias - запрет токенов; "
                   "extra_sampling - JSON-сквозняк для будущих полей движка.")

    def run(self, top_k=-1, min_p=-1.0, repeat_penalty=-1.0, repeat_last_n=-1,
            presence_penalty=-1.0, frequency_penalty=-1.0, stop="",
            logit_bias="", extra_sampling=""):
        out: dict = {}
        if top_k is not None and int(top_k) >= 1:
            out["top_k"] = int(top_k)
        if min_p is not None and 0 <= min_p <= 1:
            out["min_p"] = float(min_p)
        if repeat_penalty is not None and repeat_penalty > 0:
            out["repeat_penalty"] = float(repeat_penalty)
        if repeat_last_n is not None and int(repeat_last_n) >= 1:
            out["repeat_last_n"] = int(repeat_last_n)
        if presence_penalty is not None and presence_penalty >= 0:
            out["presence_penalty"] = float(presence_penalty)
        if frequency_penalty is not None and frequency_penalty >= 0:
            out["frequency_penalty"] = float(frequency_penalty)
        stops = [s.strip() for s in str(stop or "").splitlines() if s.strip()]
        if len(stops) > 4:
            print(f"[PolyStrata] sampling: stop-строк {len(stops)} > 4 - взяты первые 4", flush=True)
            stops = stops[:4]
        if stops:
            out["stop"] = stops
        bias = _parse_logit_bias(logit_bias)
        if bias:
            out["logit_bias"] = bias
        extra = _parse_extra_sampling(extra_sampling)
        if extra:
            out["extra"] = extra
        shown = ", ".join(f"{k}={v}" for k, v in sorted(out.items())) or "все не заданы (умолчания сервера)"
        print(f"[PolyStrata] sampling: {shown}", flush=True)
        return (out,)


# ================================================================ нода: Memory
class PolyStrataMemory:
    """Обработчик истории чата (хендлер, как в llama.cpp-нодах). Сам по себе никуда не
    ходит: хранит пары вопрос-ответ, а нода Chat (вход 'memory') дописывает их в каждый
    запрос. Без подключения Chat работает одиночным запросом.

    Одна Memory на несколько нод Chat = общая история (один разговор). Экземпляр ноды
    хранит историю между запусками workflow; 'clear: True' чистит её в этом запуске.
    В историю пишется только текст: картинки/кадры текущего запроса модели уходят,
    но в истории не сохраняются.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_turns": ("INT", {"default": 10, "min": 0, "max": 500, "step": 1,
                                      "tooltip": "Сколько последних ходов (вопрос+ответ) держать в истории. "
                                                 "0 = без ограничения (следите за контекстом модели)."}),
                "system_prompt_preset": (system_prompt_options(),
                                         {"default": NO_SYSTEM_PROMPT,
                                          "tooltip": "Пресет системного промпта из ComfyUI/models/LLM/prompts "
                                                     "(.txt, можно в подпапках) - тот же путь, что в llama.cpp-нодах. "
                                                     "none - без пресета. Выбранный пресет ПЕРЕКРЫВАЕТ поле "
                                                     "system_prompt_text. Список обновляется при каждой перезагрузке "
                                                     "страницы ComfyUI."}),
                "system_prompt_text": ("STRING", {"multiline": True, "default": "",
                                                  "tooltip": "Ручной системный промпт этого разговора; перекрывает "
                                                             "системный промпт ноды Chat, если задан. "
                                                             "Используется, если пресет = none."}),
                "clear": ("BOOLEAN", {"default": False,
                                      "tooltip": "True - очистить историю в начале этого запуска (разговор "
                                                 "с чистого листа, затем запишется новый ход)."}),
            },
        }

    RETURN_TYPES = (PS_MEMORY,)
    RETURN_NAMES = ("memory",)
    FUNCTION = "run"
    CATEGORY = "PolyStrata"
    DESCRIPTION = ("История чата для ноды PolyStrata Chat: накопление хода за ходом, лимит ходов, "
                   "системный промпт разговора, очистка.")

    def __init__(self):
        self._messages: list = []      # [{"role": "user"/"assistant", "content": str}, ...]
        self._max_turns = 10
        self._system = ""

    def run(self, max_turns=10, system_prompt_preset=NO_SYSTEM_PROMPT, system_prompt_text="",
            clear=False):
        self._max_turns = int(max_turns)
        self._system = resolve_system_prompt(system_prompt_preset, system_prompt_text)
        if clear and self._messages:
            n = len(self._messages) // 2
            self._messages = []
            print(f"[PolyStrata] память: история очищена ({n} ход(ов))", flush=True)
        return (self,)

    # ---------------------------------------------------------------- API для Chat
    def history(self) -> list:
        return list(self._messages)

    def turns(self) -> int:
        return len(self._messages) // 2

    def system_line(self) -> str:
        return self._system

    def remember(self, user_text: str, reply_text: str) -> int:
        """Дописать ход (вопрос+ответ) и подрезать историю до max_turns."""
        self._messages.append({"role": "user", "content": str(user_text or "")})
        self._messages.append({"role": "assistant", "content": str(reply_text or "")})
        if self._max_turns and self._max_turns > 0:
            keep = int(self._max_turns) * 2
            if len(self._messages) > keep:
                self._messages = self._messages[-keep:]
        return self.turns()


# ================================================================ нода: Unload
class PolyStrataUnload:
    """Полная выгрузка модели (POST /v1/unload): движок останавливается, освобождается
    ВСЯ VRAM (и RAM модели) - максимум памяти для ComfyUI. Сервер, порт и API живы.

    Больше ничего: никакого частичного резерва (для этого есть reserve_mib в ноде Chat).
    Заодно: сбрасывает запомненный на сервере reserve_mib к стартовому (после unload
    модель проснётся с ПОЛНЫМ кэшем экспертов) и ВОЗВРАЩАЕТ ComfyUI исходное
    EXTRA_RESERVED_VRAM - бронь, выставленная нодой Chat, снимается.

    Следующий чат-запрос разбудит модель САМ (тёплый старт за секунды), поэтому ноды
    Load в паке нет и не нужно.

    Сквозная: any_input -> any_output, чтобы строить непрерывную цепочку.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("STRING", {"default": "127.0.0.1:8080", "tooltip": SERVER_TOOLTIP}),
                "wait_s": ("INT", {"default": 0, "min": 0, "max": 300, "step": 5,
                                   "tooltip": "Сколько секунд ждать завершения активного запроса "
                                              "(до 300). 0 - если сервер чем-то занят, будет отказ 409 "
                                              "model_busy; поставьте 30-60, если генерация может ещё идти."}),
                "api_key": ("STRING", {"default": "", "tooltip": API_KEY_TOOLTIP}),
            },
            "optional": {
                "any_input": (ANY, {"tooltip": "Опционально: сквозной вход (вывод чат-ноды и т.п.) - задает "
                                               "момент срабатывания и прокидывается на выход."}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("any_output",)
    FUNCTION = "run"
    CATEGORY = "PolyStrata/memory"
    OUTPUT_NODE = True
    DESCRIPTION = ("Полностью выгрузить модель PolyStrata: вся VRAM и RAM освобождены, резерв "
                   "сброшен, EXTRA_RESERVED_VRAM ComfyUI восстановлен, сервер жив; следующий "
                   "запрос чата грузит модель сам. Сквозной вход/выход.")

    def run(self, server, wait_s=0, api_key="", any_input=None):
        client = PolyStrataClient(server, api_key)
        try:
            out = client.unload(float(wait_s), stop_check=_throw_if_interrupted)
        except _Interrupted:
            print("[PolyStrata] ПРЕРВАНО (Cancel ComfyUI) во время выгрузки", flush=True)
            raise
        st = out.get("status")
        if st == "unloaded":
            msg = ("Модель выгружена полностью: движок остановлен, ВСЯ VRAM и RAM модели "
                   "освобождены.\nСервер, порт и API живы. Следующий чат-запрос разбудит модель "
                   "САМ (тёплый старт, секунды).")
        elif st == "not loaded":
            msg = "Модель уже спит."
        elif st == "unsupported":
            raise PolyStrataError("Движок этого сервера не поддерживает unload (см. лог PolyStrata).")
        else:
            raise PolyStrataError(f"Неожиданный статус unload: {st} - {str(out)[:300]}")
        # reserve_mib, запомненный сервером, переживает unload (применится при загрузке):
        # сбрасываем к стартовому, чтобы модель проснулась с полным кэшем.
        try:
            r = client.vram(None)
            if r.get("status") == "not loaded":
                msg += "\nРезерв VRAM сброшен к стартовому (применится с загрузкой полный кэш)."
            else:
                msg += "\nРезерв VRAM сброшен к стартовому."
        except PolyStrataError as e:
            msg += f"\nРезерв VRAM сбросить не удалось ({e}) - модель проснётся с прежним резервом."
        # ComfyUI: вернуть исходное EXTRA_RESERVED_VRAM (модель больше не держит VRAM)
        comfy_note = update_global_vram_reservation(0)
        msg += f"\nComfyUI: {comfy_note}."
        print(f"[PolyStrata] unload (wait_s={wait_s}):\n{msg}", flush=True)
        return {"ui": {"text": [msg]}, "result": (any_input,)}


# ================================================================ регистрация
NODE_CLASS_MAPPINGS = {
    "PolyStrataChat": PolyStrataChat,
    "PolyStrataSampling": PolyStrataSampling,
    "PolyStrataMemory": PolyStrataMemory,
    "PolyStrataUnload": PolyStrataUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PolyStrataChat": "PolyStrata Chat",
    "PolyStrataSampling": "PolyStrata Sampling",
    "PolyStrataMemory": "PolyStrata Memory",
    "PolyStrataUnload": "PolyStrata Unload (полная выгрузка)",
}

# Авто-выгрузка модели (механика llama.cpp-нод): хуки на каждый запуск воркфлоу и на
# unload_all_models. В тестах/без ComfyUI тихо возвращается False.
try:
    install_auto_unload_hooks()
except Exception as _e:
    print(f"[PolyStrata] авто-выгрузка: не удалось установить хуки ({_e})", flush=True)
