"""ComfyUI-PolyStrata - ноды для сервера PolyStrata (https://github.com/VecSzn/PolyStrata).

Сервер запускается пользователем отдельно; ноды шлют запросы на его OpenAI-совместимый API
и управляющие эндпоинты (/v1/vram, /v1/unload, /v1/status).

Ноды (v8): Chat - SCHEMA V3 (comfy_api.latest) с динамическими autogrow-входами картинок
(img_0, img_1, ...) и ОТДЕЛЬНО видео (vid_0, vid_1, ...), каждый vid-пин = отдельное видео:
блок [Video N] + кадры по порядку, video_fps (в субноде Media) вставляет между кадрами
метки времени '<X.X seconds>' (формат llama.cpp при декоде видео); temperature/top_p/
top_k/reasoning/seed - только в главной ноде, без дубликатов в Sampling. Выходы: text +
thoughts. Кадры заранее подгоняются под контекст движка (двигок запросы не обрезает).
Тоггл clear_comfy_vram: перед генерацией выгрузить модели САМОГО ComfyUI (как кнопка
'Free model and node cache') - в одном запуске воркфлоу ОДИН РАЗ: лист картинок на входе,
цикл list-паков, перебор сидов или несколько нод Chat исполняют ноду многократно, но
после первой очистки выгружать нечего (результат тот же) - повторные пропускаются; флаг
сбрасывает хук PromptExecutor.execute в начале каждого запуска. Листы на любых входах
(list-паки) обрабатываются безопасно и без перечисления имён виджетов (универсальный
нормализатор всего kwargs): скалярные виджеты берут первый элемент, списки
картинок разворачиваются в пины.
Компактный интерфейс (v0.8.0): виджетов api_key/extra_mib НЕТ - доступ localhost,
резерв - один reserve_mib; лимиты кадров (max_frame_side/max_images/max_video_frames/
video_fps) - в опциональной субноде Media (вход 'media'; без неё - дефолты 1024/16/8/0).

Смена модели (v0.9.0): виджет model в ноде Chat - дропдаун установленных моделей сервера
(GET /api/models - реестр strata-*.json; перечитывается при КАЖДОМ построении ноды,
поэтому F5 / R в ComfyUI дают свежий список, кеш 15 c). Сентинел 'как на сервере' -
не влиять (поле 'model' в запрос не пишется, как в v0.8.0); выбранная модель, если не
текущая, ПЕРЕКЛЮЧАЕТ сервер до генерации (POST /api/models/load): текущий движок
останавливается, новая грузится по СВОЕМУ конфигу strata-<модель>.json.
Interrupt (кнопка Cancel ComfyUI) прерывает генерацию: interruptible-запрос идёт через
http.client, при срабатывании stop_check сокет закрывается ПО-НАСТОЯЩЕМУ (FIN) - watch-поток
PolyStrata (_watch_client) видит EOF и отменяет генерацию движка
(comfy.model_management.throw_exception_if_processing_interrupted поллится во время запроса;
InterruptProcessingException наследует BaseException и не глотается).
Sampling - остальной сэмплинг: top_k, min_p, штрафы, stop, logit_bias (запрет токенов),
extra_sampling (JSON-сквозняк на будущие поля движка).
Memory - история. Unload - полная выгрузка, сквозная; возвращает ComfyUI исходное
EXTRA_RESERVED_VRAM. Reserve VRAM в Chat пишет фактическое потребление PolyStrata + extra_mib
в EXTRA_RESERVED_VRAM и headroom comfy-aimdo (метод update_global_vram_reservation из
llama.cpp-нод) - ComfyUI грузит свои модели без OOM.

На ComfyUI без comfy_api нода Chat автоматически собирается в V1-варианте (входы images/videos
фиксированные) - поведение то же.

Важно: elastic VRAM у strata-poly ВСТРОЕН (движок 0.1.45-poly.6+, обновление - UPDATE.bat).
Ключ "vram_elastic": true в конфиге strata-poly-модели ВРЕДЕН (движок падает на неизвестном
флаге --vram-elastic) - не добавляйте его, а если добавляли, удалите.
"""

try:
    # обычный случай: ComfyUI загружает папку как пакет (современные версии)
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # старый загрузчик ComfyUI: папка грузится как простой модуль без package-контекста.
    # Регистрируем подмодули вручную, под коллизионно-безопасными именами в sys.modules.
    import importlib.util
    import os
    import sys

    _dir = os.path.dirname(os.path.abspath(__file__))

    def _load(name: str, path: str):
        mod_name = f"_comfyui_polystrata_{name}"
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    # ps_client под обычным именем тоже регистрируем: nodes.py делает `from ps_client import ...`
    # в запасной ветке - так он разрешится в НАШ модуль, а не в чужой одноимённый
    if "ps_client" not in sys.modules:
        _spec = importlib.util.spec_from_file_location("ps_client", os.path.join(_dir, "ps_client.py"))
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["ps_client"] = _mod
        _spec.loader.exec_module(_mod)

    _nodes = _load("nodes", os.path.join(_dir, "nodes.py"))
    NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.9.0"
