# AGENTS.md - ComfyUI-PolyStrata

Инструкция для ИИ-агентов и разработчиков, правящих этот нодпак. Прочти до конца, прежде
чем менять код.

## Что это

ComfyUI custom nodes для локального сервера [PolyStrata](https://github.com/VecSzn/PolyStrata)
(форк llama.cpp: C++-движок `strata`/`strata-poly` + Python-сервер `serve/server.py`).
Сервер ЗАПУСКАЕТ ПОЛЬЗОВАТЕЛЬ (run-`<модель>`.bat); ноды - только HTTP-клиенты.
Зависимости пакета: `requests` (+ numpy/Pillow в окружении ComfyUI).

## Карта файлов

| Файл | Роль |
| --- | --- |
| `nodes.py` | Все ноды: `PolyStrataChat`, `PolyStrataSampling`, `PolyStrataMemory`, `PolyStrataUnload` + `NODE_CLASS_MAPPINGS`. Тултипы на русском, там же - основная документация виджетов. В Chat - тоггл `clear_comfy_vram` (очистка памяти моделей ComfyUI перед генерацией, один раз на запуск воркфлоу) и обработка листов на входах: универсальный `_scalars` по всему kwargs (без перечисления имён) + `_as_pins`. |
| `ps_client.py` | Тонкий HTTP-клиент (`PolyStrataClient`): chat/vram/unload/status, разбор ошибок (409 model_busy, 503+Retry-After, 400 --vram-elastic, 404 model); interruptible-запросы (`stop_check` - колбэк Cancel ComfyUI, при срабатывании соединение закрывается). |
| `__init__.py` | Загрузка двумя способами (package-спека и старый загрузчик ComfyUI без package-контекста). Версия пакета - здесь (`__version__`). |
| `tests/test_nodepack.py` | Портативный тест-харнесс: поднимает mock-сервер PolyStrata (форматы из `serve/server.py` и `docs/VRAM_ELASTIC.md`) и прогоняет все ноды. Запуск: `python tests/test_nodepack.py` (или `python3 scripts/test_nodepack.py` из workspace). |
| `worklog.md` | Журнал изменений с обоснованиями - дополняется при каждом изменении поведения. |
| `README.md` | Пользовательская документация (RU). |

## Инварианты (не ломать)

1. **Сервер запускает пользователь.** В нодах нет путей к GGUF, ngl, ctx_size и прочих
   llama.cpp-параметров загрузки - только адрес сервера + api_key.
2. **Без стриминга.** `/v1/chat/completions` всегда `stream:false`, ответ целиком.
3. **Медиа - autogrow-пины schema V3.** Нода Chat - `comfy_io.ComfyNode` (comfy_api.latest):
   autogrow `img_0..img_49` (картинки, IMAGE) и ОТДЕЛЬНО `vid_0..vid_49` (видео = IMAGE-батч
   последовательных кадров); каждый vid-пин = отдельное видео: блок «[Video N]» + кадры по
   порядку, при `video_fps` > 0 - метки времени '<X.X seconds>' перед КАЖДЫМ кадром
   (формат llama.cpp при декоде видео, mtmd-helper.cpp; время = исходный индекс кадра / fps);
   картинки - сквозная нумерация «[Image K]». Никаких входов
   типа VIDEO и аудио. На ComfyUI без comfy_api Chat собирается в V1 (фиксированные
   `images`/`videos`) - тот же формат через `_as_pins`. Выборка кадров - np.linspace (исходные
   индексы СОХРАНЯЮТСЯ - метки времени считаются от них);
   `max_images` / `max_video_frames` - **НА КАЖДЫЙ пин** своего типа (0 = все кадры пина, потолки 64/128 на пин), и кадры
   заранее подгоняются под контекст движка (`_fit_media_to_context`, окно из
   `context.native` /v1/status, запасной путь /props): двигок запросы НЕ обрезает -
   отклоняет 400 'exceeds the context window'. **Interrupt (Cancel ComfyUI)**:
   `_throw_if_interrupted()` (nodes.py) вызывается до запроса, после reserve/медиа и
   поллингом ВО ВРЕМЯ запроса (`ps_client._request_interruptible`: http.client, сокет наш;
   при срабатывании сокет ЗАКРЫВАЕТСЯ - FIN - и watch-поток сервера `_watch_client` видит
   EOF и ставит cancel: движок останавливается. НЕ на requests: Session.close() сокет
   висящего запроса не закрывает - сервер обрыва не видел и дорабатывал до конца,
   исправлено в 0.7.1).
   `InterruptProcessingException` наследует **BaseException** -
   не глотать широкими except Exception; ловить только чтобы залогировать «ПРЕРВАНО»
   и сразу re-raise. Ход в Memory при прерывании не записывается.
4. **reserve_mib = 0 не отправляет НИКАКИХ запросов** `/v1/vram` (0 на стороне сервера
   означает «держать свободной 0 MiB» - это другое; не путать) и НЕ трогает бронь ComfyUI.
5. **Частичный успех `/v1/vram` - это HTTP 200**: сверять `vram_free_mib >= reserve_mib`
   и помечать «ЧАСТИЧНОЕ ОСВОБОЖДЕНИЕ» в логе.
6. **Нод Load/VRAM в паке нет** (v3): модель грузится сервером по требованию; резерв -
   виджеты `reserve_mib`/`extra_mib` в Chat; полная выгрузка - Unload (она же сбрасывает
   запомненный сервером резерв через `{"reserve_mib": null}`).
7. **Сквозная Unload**: `any_input` (тип `*`, `AnyType.__ne__ -> False`) → `any_output`,
   `OUTPUT_NODE = True`.
8. **Дубликатов сэмплинга нет**: `temperature`, `top_p`, `reasoning`, `seed` - ТОЛЬКО в ноде
   Chat (в Sampling их нет; seed с control_after_generate, 0 = случайный - не отправляется).
   Sampling - остальное: top_k, min_p, repeat_penalty, repeat_last_n,
   presence/frequency_penalty, stop + logit_bias + extra_sampling.
9. **Семплеры - только те, что реально принимает сервер.** Белый список (см. ниже) + `stop`
   (до 4 строк) + reasoning. НИКАКИХ виджетов typical_p/tfs/xtc/dry/dynatemp/mirostat/
   smoothing/top_n_sigma: сервер их молча отбрасывает, мёртвый контрол - баг. На будущее -
   extra_sampling (JSON-сквозняк в тело запроса), а не мёртвые виджеты.
10. **Sentinel «не задано» = -1** (для stop/logit_bias/extra_sampling - пустая строка) в ноде
    Sampling; клиент шлёт только заданные поля. Штрафы уходят в канонических именах сервера:
    `repetition_penalty`, `penalty_last_n` (не алиасы llama.cpp).
11. **Бронь VRAM для ComfyUI - метод llama.cpp-нод** (update_global_vram_reservation из
    LlamaCPP-Subprocess): reserve > 0 → в `comfy.model_management.EXTRA_RESERVED_VRAM`
    пишется ФАКТИЧЕСКОЕ потребление PolyStrata (`/v1/status machine.gpu.total_mib -
    vram_free_mib` из ответа /v1/vram) + extra_mib (НЕ запрошенная цель!), + headroom в
    comfy-aimdo (`comfy_aimdo.control.set_simple_vram_headroom`), если aimdo активен;
    исходное значение кэшируется один раз. У ноды Unload / на спящей модели - исходное
    значение ВОЗВРАЩАЕТСЯ. Ошибки брони - best-effort, генерацию не роняют.
12. **Ошибка резерва не роняет генерацию** (ловится, пишется «РЕЗЕРВ НЕ ПРИМЕНЁН»);
    ошибка чата - роняет (PolyStrataError).
13. Ключ `"vram_elastic": true` в конфиге strata-poly-модели **вреден** (движок падает на
    неизвестном флаге `--vram-elastic`): тултипы и ошибки должны говорить «удалите ключ»,
    никогда - «добавьте».
14. **Очистка памяти ComfyUI - ОДИН РАЗ на запуск воркфлоу** (тоггл `clear_comfy_vram` в
    Chat): `free_comfy_model_memory` = `mm.unload_all_models()` + `soft_empty_cache` + gc
    (механика кнопки 'Free model and node cache'); модели PolyStrata не задеваются (хук
    авто-выгрузки внутри выполнения нод серверы не трогает). Флаг `_COMFY_CLEAR` ставится на
    первом исполнении ноды и сбрасывается `note_workflow_start()` из хука
    `PromptExecutor.execute` (тот же хук, что у авто-выгрузки): лист картинок, цикл
    list-пака, перебор сидов, несколько нод Chat исполняют ноду многократно - очистка одна
    (после первой выгружать нечего, результат тот же). Хуки не установились (comfy
    недоступен) - очистка на КАЖДОМ вызове (никогда не «залипает»).
15. **Листы на входах обрабатываются безопасно и БЕЗ хардкода имён виджетов**: list-паки
    циклов/переборов могут прислать список в ЛЮБОЙ вход. Нормализация универсальна:
    execute/generate принимают **kw и прогоняют ВЕСЬ kwargs через `_scalars` (каждый
    список -> первый элемент, None для пустого; имена не перечисляются - новый виджет
    защищён автоматически). Единственное исключение - `_LIST_INPUTS` (images/videos):
    там список и есть значение, его разворачивает в пины `_as_pins` (батч остаётся
    пином, нумерация кадров сквозная); memory/sampling-списки берут первый элемент -
    история не теряется молча. Ядро `_chat_generate` получает уже нормализованные
    скаляры - НЕ возвращать per-name `_scalar` в ядро при добавлении виджетов.

## Проверенные факты об API PolyStrata (обновляй по репозиторию сервера)

Источник истины - репозиторий VecSzn/PolyStrata:

- **Сэмплинг**: `serve/server.py` → `check_request_sampling` (~L7511: типы и алиасы
  `repeat_penalty→repetition_penalty`, `repeat_last_n→penalty_last_n`) и
  `StrataEngine.sampling_keys` (~L1203: что реально уходит движку: temperature, top_p,
  top_k (сжимается до 1..64), min_p, penalty_repeat/freq/present, penalty_last_n, seed,
  image_pad, strata_tune/ckpt/img_keys/pin, cvec, **logit_bias** через `bias_api.engine_key`).
  `logit_bias` читается из КОРНЯ запроса (`_openai`, ~L6530: `req.get("logit_bias")`),
  формат - `serve/logit_bias.py`: dict {"token_id": bias} или список [[id, bias]],
  bias -100..100 (−100 = запрет; в списке `false` тоже −100), токен < vocab_size.
  xtc/dry/dynatemp/mirostat/tfs_z/typical_p/smoothing/top_n_sigma - 0 упоминаний
  (перепроверено по свежему main 2026-10-31).
  `stop` - `stop_strings(req)` (до `OPENAI_MAX_STOP`=4). Thinking - ТОЛЬКО два механизма,
  которые сервер реально читает (serve/frontend.py `openai_to_messages` + `effort_kwargs`):
  `reasoning_effort` топ-левел ("none"→thinking off; low|medium|high→xhigh) и
  `chat_template_kwargs: {"enable_thinking": false}`; топ-левел `enable_thinking` из тела
  запроса сервер НЕ читает - при пресете none нода Chat шлёт ОБА механизма.
- **Schema V3 ComfyUI**: образец - нодпак пользователя bananasss00/ComfyUI-LlamaCPP-Subprocess
  (`/tmp/comfy-llamacpp/__init__.py`): `from comfy_api.latest import io`, класс
  `io.ComfyNode`, `define_schema` → `io.Schema(node_id/display_name/category/description/
  inputs/outputs)`, autogrow - `io.Autogrow.TemplatePrefix(io.Image.Input("img", optional=True),
  prefix="img", min=0, max=50)` + `io.Autogrow.Input("images", template=...)`; autogrow-вход
  приходит в execute СЛОВАРЁМ {"img_0": тензор}; регистрация V3-класса - прямо в
  NODE_CLASS_MAPPINGS (V1-ноды рядом - норм). ВРЕМЕННО: V1-фолбэк при отсутствии comfy_api.
- **VRAM**: `docs/VRAM_ELASTIC.md`. `{"reserve_mib": N}` = сколько MiB держать СВОБОДНЫМИ
  на карте (считает всё, не только PolyStrata); `null` = стартовый резерв движка; `0` =
  «не держать ничего свободным» (отдать модели всё). Сервер помнит резерв в памяти и
  пере-применяет после рестарта движка; на спящей модели только записывает («applied when
  the model loads»). Пол: 128 слотов + smallest prompt-path loan. Сегмент по умолчанию 512 MiB.
- **Unload**: `POST /v1/unload?wait=N` - `wait` 0/нет = сразу (409 model_busy если занято),
  1/true = максимум по умолчанию, число = столько секунд (до 3600).
- **Медиа**: JPEG/PNG/BMP/GIF уходят как есть, остальное конвертируется сервером в PNG;
  на нашей стороне - JPEG base64 data-URI, длинная сторона ≤ `max_frame_side` (виджет,
  по умолчанию 1024; 0 = без ужатия). Оценка vision-токенов кадра - `_frame_tokens`:
  ceil(H/28)*ceil(W/28) + 12 (у Qwen-VL 28 px/токен; у Qwen3-VL фактически 32 - оценка
  консервативная, с запасом; +12 покрывает метки '[Image K]'/'[Video N]'/«<X.X seconds>»).
- **Окно контекста**: `/v1/status` → `context.native` (или `cache_max_tokens`); запасной
  путь `/props` → `default_generation_settings.n_ctx`. Кэш по адресу сервера + времени
  старта движка (`_SERVER_CTX_CACHE`). 0 = недоступно - префлайт отключается.
- **max_tokens**: 0/отсутствие = «без ограничения» (сервер сам считает по контексту).
- **Статус**: `GET /v1/status` → `vram.expert_cache_mib/expert_cache_full_mib/
  expert_slots/vram_free_mib/elastic`, `model`, `loaded`, `vision.enabled`,
  `machine.gpu.total_mib/used_mib` (нужно для брони EXTRA_RESERVED_VRAM).
- **EXTRA_RESERVED_VRAM в ComfyUI core**: `comfy/model_management.py` ~L877 (400/600 MiB
  базово, `--reserve-vram` переопределяет); используется в `load_models_gpu` (~L944-948:
  `extra_mem` и `minimum_memory_required`) - управление оффлоадом и решением
  полная/lowvram-загрузка. Писать туда = «запас под чужую VRAM».
- **Видео (проверено по коду Jan + llama.cpp, 2026-10-31)**: Jan (v0.8.3+) шлёт видеофайл
  ЦЕЛИКОМ частью `{"type": "input_video", "input_video": {"data": "<base64>"}}`
  (web-app/src/lib/video-sentinel.ts + model-factory.ts `decodeVideoSentinelsInBody`; алиас
  `video_url` принимает llama.cpp tools/server/server-common.cpp ~L1305) - а движок сам
  декодирует через ffmpeg: кадры + метки '<X.X seconds>' перед каждой темпоральной группой
  (tools/mtmd/mtmd-helper.cpp ~L1044: `snprintf(ts_buf, "<%.1f seconds>", ts_s)`;
  второй режим - '[%dm%.2fs]' каждые 5 с, ~L1063). Полифорк сервера НЕ умеет input_video:
  в serve/server.py - 0 упоминаний video (принимаются только image_url/Anthropic image).
  Поэтому нода повторяет разметку движка сама: блок '[Video N]' (НЕ 'sequential frames' -
  это провоцировало перечисление кадров) + '<X.X seconds>' от video_fps. Если PolyStrata
  добавит input_video - можно слать файл, но ComfyUI отдаёт батч кадров, не файл.
  В связке Jan→PolyStrata видео НЕ доходит вовсе: input_video-часть сервер выбрасывает
  (serve/frontend.py `_check_parts` - непрочитанные части «left out of the prompt»,
  IMAGE_PARTS = image_url/input_image/image), подсказок-текста Jan к видео не добавляет -
  так что кадры с метками - единственный путь (перепроверено по свежим исходникам
  janhq/jan main + VecSzn/PolyStrata main 2026-10-31).

Если меняешь набор семплеров/эндпоинтов - сначала перепроверь эти места в репозитории
сервера и обнови: `ps_client.py` (докстринг!), тултипы `nodes.py`, таблицу в `README.md`,
инварианты выше и тесты.

## Запуск тестов

```bash
python tests/test_nodepack.py      # 170 проверок против mock-сервера
```

Все проверки должны быть зелёными перед коммитом. Mock-сервер в тесте - образец форматов
ответов: правь его вместе с изменениями клиента.

## Стиль

- Тултипы и сообщения - на русском, конкретные (что сделает, что вернёт, что означает 0/-1).
- Названия полей в payload - канонические имена сервера PolyStrata, не llama.cpp.
- Никаких новых зависимостей: requests/numpy/PIL уже есть в ComfyUI.
- Версия пакета: `__init__.py.__version__` + `pyproject.toml` + заголовок README - синхронно.
- После изменения поведения - запись в `worklog.md` (дата, что, почему, пруфы).
