"""ps_client.py - HTTP-клиент сервера PolyStrata для ComfyUI-нод.

Сервер запускается пользователем отдельно (`polystrata serve`, run-<модель>.bat); ноды только
обращаются к его OpenAI-совместимому API и к управляющим эндпоинтам PolyStrata:

  POST /v1/chat/completions   генерация (без стриминга; картинки - OpenAI image_url части)
                              сэмплинг - ПОЛНЫЙ список принятых полей (serve/server.py,
                              check_request_sampling + StrataEngine.sampling_keys):
                              temperature, top_p, top_k (int, движок сжимает до 1..64), min_p,
                              repetition_penalty (алиас repeat_penalty), penalty_last_n (алиас
                              repeat_last_n, только >0), presence_penalty, frequency_penalty,
                              stop (строка или список до 4), seed (int >0), logit_bias
                              ({"<token_id>": bias -100..100, -100 = запрет токена}) +
                              reasoning: reasoning_effort ("none" = thinking off; low/medium/high,
                              "high" сервер поднимает до xhigh) и chat_template_kwargs
                              {"enable_thinking": false} (конвенция vLLM/llama.cpp). ВАЖНО: эти два
                              механизма сервер РЕАЛЬНО читает (serve/frontend.py openai_to_messages),
                              а топ-левел поле "enable_thinking" из тела запроса - НЕТ (молча
                              игнорируется). Прочие семплеры llama.cpp (typical_p, tfs_z, xtc, dry,
                              dynatemp, mirostat, smoothing_factor, top_n_sigma) сервер отбрасывает.
  POST /v1/vram               {"reserve_mib": N} - держать N MiB VRAM свободными (цель по
                              свободной VRAM на карте; частичный успех = 200, сверять
                              vram_free_mib >= reserve_mib), {"reserve_mib": null} - вернуть
                              резерв к стартовому. Сервер запоминает резерв и сам применяет
                              его после перезапуска движка; на спящей модели только записывает.
                              (У strata-poly встроено с 0.1.45-poly.6: без ключей в конфиге.)
  POST /v1/unload?wait=N      выгрузить модель: движок остановлен, вся VRAM освобождена,
                              сервер и порт живы; следующий запрос сам запустит тёплый старт
  GET  /v1/status             состояние: model, loaded, vram{..., elastic}, machine.gpu,
                              context.native (окно движка в токенах)
  GET  /props                 свойства сервера (default_generation_settings.n_ctx и пр.)

Прерывание (кнопка Cancel / interrupt в ComfyUI): методы chat/vram/unload принимают
stop_check - колбэк comfy.model_management.throw_exception_if_processing_interrupted.
Interruptible-запросы идут через http.client (НЕ requests): сокет наш (conn.sock),
стоп-колбэк поллится каждые 0.25 c, и при срабатывании он закрывается ПО-НАСТОЯЩЕМУ -
ОС отправляет FIN, и watch-поток PolyStrata (_watch_client в serve/server.py, #430 #431:
select + recv(MSG_PEEK) каждые 0.5 c, EOF = клиент ушёл) ставит cancel - движок
останавливается ('client went away: stop the engine'), слот освобождается. ВАЖНО:
requests.Session.close() сокет висящего запроса НЕ закрывает (соединение не вернулось
в пул) - сервер обрыва не видел и дорабатывал до конца; поэтому interruptible-путь
переведён на http.client (v0.7.1). Исключение InterruptProcessingException (наследует
BaseException!) летит наверх - ComfyUI останавливает выполнение сам.
"""
from __future__ import annotations

import http.client
import json
import select
import socket
import time
from urllib.parse import urlencode, urlsplit

import requests

DEFAULT_PORT = 8080


class PolyStrataError(RuntimeError):
    """Ошибка обращения к серверу PolyStrata (сеть, HTTP-статус, логика сервера)."""


def _server_base(server: str) -> str:
    """'  http://127.0.0.1:8080/ ' -> 'http://127.0.0.1:8080'; без порта добавляется 8080."""
    s = (server or "").strip()
    for prefix in ("https://", "http://"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.rstrip("/")
    if not s:
        raise PolyStrataError("Адрес сервера PolyStrata пуст - укажите IP:порт, например 127.0.0.1:8080")
    if ":" not in s:                      # только IP/хост - подставляем порт по умолчанию
        s = f"{s}:{DEFAULT_PORT}"
    return f"http://{s}"


def _server_display(server: str) -> str:
    try:
        return _server_base(server).removeprefix("http://")
    except PolyStrataError:
        return str(server)


def _error_message(status: int, text: str) -> str:
    """Человекочитаемый текст из тела ошибки PolyStrata ({\"error\": {\"message\": ...}})."""
    try:
        body = json.loads(text)
    except ValueError:
        t = (text or "").strip()
        return t[:400] if t else f"HTTP {status}"
    err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or body)[:400]
    if isinstance(err, str):
        return err[:400]
    return json.dumps(body, ensure_ascii=False)[:400]


class PolyStrataClient:
    """Тонкий клиент одного сервера PolyStrata (ip:порт + опциональный api_key)."""

    def __init__(self, server: str, api_key: str = ""):
        self.base = _server_base(server)
        self.api_key = (api_key or "").strip()
        # (connect, read); генерация больших моделей бывает долгой - read щедрый
        self.timeouts = {"chat": (10, 1800), "control": (10, 360), "quick": (10, 30)}

    # ------------------------------------------------------------------ низкий уровень
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
            h["x-api-key"] = self.api_key
        return h

    def _request(self, method: str, path: str, payload=None, params=None,
                 timeout=(10, 300), retry_503: int = 0, stop_check=None) -> dict:
        """Один запрос; 503 с Retry-After (min_free_vram_mib не выполнен) повторяется retry_503 раз.
        stop_check - колбэк прерывания (Cancel ComfyUI): вызывается до запроса, поллингом во время
        него и между повторами 503; при срабатывании сокет закрывается ПО-НАСТОЯЩЕМУ (FIN - сервер
        отменяет генерацию), исключение летит наверх."""
        url = self.base + path
        attempts = max(0, int(retry_503)) + 1
        for attempt in range(attempts):
            if stop_check is not None:
                stop_check()
            try:
                if stop_check is None:
                    resp = requests.request(
                        method, url,
                        json=payload if payload is not None else ({}) if method == "POST" else None,
                        params=params, headers=self._headers(), timeout=timeout)
                    status, text = resp.status_code, resp.text
                    hdrs = {k.lower(): v for k, v in resp.headers.items()}
                else:
                    status, text, hdrs = self._request_interruptible(
                        method, url, payload, params, timeout, stop_check)
            except requests.exceptions.ConnectionError as e:
                raise PolyStrataError(
                    f"Не удалось подключиться к PolyStrata на {self.base} - сервер запущен? "
                    f"Запустите его сами (`polystrata serve` или run-<модель>.bat) и проверьте адрес/порт. ({e})"
                ) from e
            except requests.exceptions.Timeout as e:
                raise PolyStrataError(
                    f"Тайм-аут запроса к {self.base}{path}: сервер не ответил за {timeout[1]} c. "
                    f"Возможно, модель ещё грузится или генерация очень долгая."
                ) from e
            except requests.exceptions.RequestException as e:
                raise PolyStrataError(f"Ошибка запроса к {self.base}{path}: {e}") from e

            if status < 400:
                if not text:
                    return {}
                try:
                    return json.loads(text)
                except ValueError:
                    return {"raw": text}

            # ---- ошибки ----
            if status == 503 and attempt < attempts - 1:
                retry_after = hdrs.get("retry-after")
                delay = min(float(retry_after), 120.0) if (retry_after or "").replace(".", "", 1).isdigit() else 5.0
                print(f"[PolyStrata] 503: {_error_message(status, text)} - повтор через {delay:.0f} c "
                      f"({attempt + 1}/{attempts - 1})", flush=True)
                self._sleep_interruptible(delay, stop_check)
                continue
            self._raise_http(status, text, path)
        raise PolyStrataError("неожиданное состояние запроса")

    def _raise_http(self, status: int, text: str, path: str):
        msg = _error_message(status, text)
        if status == 400 and "--vram-elastic" in msg:
            raise PolyStrataError(
                "Сервер передал движку флаг --vram-elastic: в конфиге модели (strata-<модель>.json) "
                "стоит \"vram_elastic\": true. Для движка strata-poly этот ключ НЕ нужен (elastic VRAM "
                "встроен, начиная с 0.1.45-poly.6) и только ломает старт движка ('unknown option'): "
                "удалите ключ из конфига и запустите модель заново; после этого ноды VRAM заработают. "
                "Сервер ответил: " + msg)
        if status == 400 and "exceeds the context window" in msg:
            raise PolyStrataError(
                "Запрос НЕ влезает в контекст модели - двигок такие запросы никогда не обрезает, "
                "только отклоняет. Главный пожиратель - медиа: каждый кадр стоит ~ (высота/28) * "
                "(ширина/28) токенов у Qwen-VL (кадр 1280x720 ~ 1200 токенов), лимиты "
                "max_video_frames / max_images в ноде Chat действуют НА КАЖДЫЙ пин своего типа. "
                "Уменьшите их или max_frame_side (длинную сторону кадра), ходы памяти (max_turns "
                "ноды Memory) или длину промпта. Сервер ответил: " + msg)
        if status == 409:
            raise PolyStrataError(
                f"Сервер занят (409 model_busy): идёт генерация или запрос в очереди - {msg}. "
                f"Для /v1/unload включите 'wait', для /v1/vram повторите после завершения генерации.")
        if status == 503:
            raise PolyStrataError(
                f"Сервер отвечает 503: {msg}. Частая причина - не выполнен min_free_vram_mib (свободной VRAM "
                f"меньше минимума, модель не стартует; освободите VRAM и повторите).")
        if status == 404 and "model not found" in msg:
            raise PolyStrataError(
                f"404: {msg}. Имя модели в ноде не совпадает с запущенной на сервере "
                f"(очистите поле 'model' или возьмите имя из GET /v1/status).")
        if status in (401, 403):
            raise PolyStrataError(f"{status}: требуется API-ключ сервера - {msg}")
        raise PolyStrataError(f"HTTP {status} от {path}: {msg}")

    # -------------------------------------------------------- прерывание (Cancel ComfyUI)
    @staticmethod
    def _sleep_interruptible(delay: float, stop_check=None) -> None:
        """sleep, прерываемый stop_check (Cancel не должен ждать Retry-After)."""
        if stop_check is None:
            time.sleep(max(0.0, float(delay)))
            return
        end = time.monotonic() + max(0.0, float(delay))
        while True:
            stop_check()
            remain = end - time.monotonic()
            if remain <= 0:
                return
            time.sleep(min(0.25, remain))

    def _request_interruptible(self, method: str, url: str, payload, params,
                               timeout, stop_check):
        """Блокирующий запрос с НАСТОЯЩИМ обрывом TCP при Cancel ComfyUI.
        Возвращает (status, text, headers_lowercase).

        requests отмену не умеет, а Session.close() НЕ закрывает сокет запроса, который
        ещё ждёт ответ (соединение «на руках» у вызова, в пул не вернулось) - сервер
        обрыва не видел и дорабатывал до конца (то, что наблюдал пользователь). Поэтому
        здесь http.client: сокет наш (conn.sock), и алгоритм такой:
          1) заголовки + тело (кусками по 256 KiB, между кусками - stop_check);
          2) ожидание ответа: каждые 0.25 c stop_check() + select по сокету;
          3) stop_check сработал -> finally закрывает сокет: ОС отправляет FIN, и
             watch-поток PolyStrata (_watch_client, serve/server.py: select + MSG_PEEK
             каждые 0.5 c) видит EOF и ставит cancel - движок останавливается
             ('client went away: stop the engine'), слот освобождается.
        InterruptProcessingException (наследует BaseException!) НЕ глотается - летит
        наверх; ошибки сети/тайм-аутов мапятся в те же сообщения, что и у requests."""
        u = urlsplit(url)
        path = u.path or "/"
        if params:
            path += ("&" if "?" in path else "?") + urlencode(params)
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        elif method == "POST":
            body = b"{}"
        connect_to, read_to = timeout
        conn = http.client.HTTPConnection(
            u.hostname, u.port or (443 if (u.scheme or "http") == "https" else 80),
            timeout=connect_to)
        try:
            conn.putrequest(method, path, skip_accept_encoding=True)
            conn.putheader("Accept", "application/json")
            if self.api_key:
                conn.putheader("Authorization", f"Bearer {self.api_key}")
                conn.putheader("x-api-key", self.api_key)
            if body is not None:
                conn.putheader("Content-Type", "application/json")
                conn.putheader("Content-Length", str(len(body)))
            conn.endheaders()                       # здесь же connect (ленивый) - сокет готов
            if body is not None:                    # тело кусками: Cancel и во время отправки
                step = 262144
                for i in range(0, len(body), step):
                    stop_check()
                    conn.send(body[i:i + step])
            while True:                             # ждём ответ: Cancel или данные
                stop_check()
                r, _, _ = select.select([conn.sock], [], [], 0.25)
                if r:
                    break
            conn.sock.settimeout(read_to)
            resp = conn.getresponse()
            text = resp.read().decode("utf-8", "replace")
            return resp.status, text, {k.lower(): v for k, v in resp.getheaders()}
        except PolyStrataError:
            raise
        except (TimeoutError, socket.timeout) as e:
            raise PolyStrataError(
                f"Тайм-аут запроса к {url}: сервер не ответил за {timeout[1]} c. "
                f"Возможно, модель ещё грузится или генерация очень долгая.") from e
        except ConnectionRefusedError as e:
            raise PolyStrataError(
                f"Не удалось подключиться к PolyStrata на {self.base} - сервер запущен? "
                f"Запустите его сами (`polystrata serve` или run-<модель>.bat) и проверьте адрес/порт. ({e})"
            ) from e
        except (OSError, http.client.HTTPException) as e:
            raise PolyStrataError(f"Ошибка запроса к {url}: {e}") from e
        finally:
            try:                                    # при interrupt - это и есть «повесить трубку»:
                conn.close()                        # FIN -> сервер отменяет генерацию
            except Exception:
                pass

    # ------------------------------------------------------------------ API нод
    def chat(self, messages: list, model: str = "", max_tokens: int = 1024, temperature=None,
             top_p=None, top_k=None, seed=None, reasoning: str = "default", min_p=None,
             repeat_penalty=None, repeat_last_n=None, presence_penalty=None,
             frequency_penalty=None, stop=None, logit_bias=None, extra=None,
             stop_check=None) -> dict:
        """POST /v1/chat/completions без стриминга; возвращает выбор целиком (dict ответа).
        None = поле не отправляется (серверные умолчания). Имена - канонические ключи
        PolyStrata: repetition_penalty, penalty_last_n (алиасы repeat_penalty/repeat_last_n
        тоже принимаются). top_k <= 0 и seed <= 0 не отправляются: 0 у движка означает
        'не задано' (top_k сжимается движком до 64). stop - список до 4 строк (OpenAI).
        logit_bias - dict {"<token_id>": bias -100..100} (OpenAI-стиль; -100 = запрет токена).
        extra - dict дополнительных полей запроса (JSON-сквозняк ноды Sampling): сливается
        в тело как есть (кроме messages/stream), перекрывая базовые значения.
        stop_check - колбэк прерывания (Cancel ComfyUI): генерация отменяется при обрыве."""
        payload = {"messages": messages, "stream": False}
        if model.strip():
            payload["model"] = model.strip()
        if max_tokens and int(max_tokens) > 0:          # 0/не задано = лимит сервера (у PolyStrata: без ограничения)
            payload["max_tokens"] = int(max_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)   # 0 - осмысленно: жадный режим
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None and int(top_k) >= 1:       # <= 0 не отправлять (движок возьмёт своё)
            payload["top_k"] = int(top_k)
        if seed is not None and int(seed) > 0:
            payload["seed"] = int(seed)
        if min_p is not None:
            payload["min_p"] = float(min_p)
        if repeat_penalty is not None and float(repeat_penalty) > 0:
            payload["repetition_penalty"] = float(repeat_penalty)   # 1 = выключен (но явный 1.0 тоже можно)
        if repeat_last_n is not None and int(repeat_last_n) >= 1:
            payload["penalty_last_n"] = int(repeat_last_n)          # 0/-1: окно движка по умолчанию
        if presence_penalty is not None and float(presence_penalty) >= 0:
            payload["presence_penalty"] = float(presence_penalty)
        if frequency_penalty is not None and float(frequency_penalty) >= 0:
            payload["frequency_penalty"] = float(frequency_penalty)
        if stop:
            stops = [s for s in stop if isinstance(s, str) and s][:4]   # сервер: до 4 строк
            if stops:
                payload["stop"] = stops
        if reasoning and reasoning != "default":
            if reasoning == "none":
                # Выключить размышления. Сервер читает reasoning_effort ("none" -> thinking off,
                # serve/frontend.py EFFORT) и chat_template_kwargs (конвенция vLLM/llama.cpp);
                # топ-левел enable_thinking из тела запроса он НЕ читает - шлём ОБА рабочих механизма.
                payload["reasoning_effort"] = "none"
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            else:                                        # low | medium | high ('high' сервер маппит в xhigh)
                payload["reasoning_effort"] = reasoning
        if logit_bias:
            payload["logit_bias"] = {str(int(k)): float(v) for k, v in dict(logit_bias).items()}
        if extra:
            for k, v in dict(extra).items():             # сквозняк; служебные поля не трогаем
                if k not in ("messages", "stream"):
                    payload[k] = v
        return self._request("POST", "/v1/chat/completions", payload, timeout=self.timeouts["chat"],
                             stop_check=stop_check)

    def vram(self, reserve_mib, stop_check=None) -> dict:
        """POST /v1/vram: reserve_mib int (>=0) - держать столько MiB VRAM свободными; None - вернуть как на старте.
        У strata-poly встроено (0.1.45-poly.6+); 400 про --vram-elastic означает лишний ключ
        \"vram_elastic\": true в конфиге модели (см. _raise_http)."""
        payload = {"reserve_mib": None if reserve_mib is None else int(reserve_mib)}
        return self._request("POST", "/v1/vram", payload, timeout=self.timeouts["control"],
                             stop_check=stop_check)

    def unload(self, wait_s: float = 0.0, stop_check=None) -> dict:
        """POST /v1/unload[: wait=<c>] - полностью выгрузить модель. wait_s>0: ждать активные
        запросы до wait_s секунд. Пробуждение - следующим чат-запросом (POST /v1/load нодам
        не нужен: модель грузится по требованию)."""
        params = {"wait": wait_s} if wait_s and float(wait_s) > 0 else None
        return self._request("POST", "/v1/unload", {}, params=params, timeout=self.timeouts["control"],
                             stop_check=stop_check)

    def status(self) -> dict:
        """GET /v1/status - состояние сервера (model, loaded, vram.elastic, machine.gpu,
        context.native - окно движка в токенах ...)."""
        return self._request("GET", "/v1/status", timeout=self.timeouts["quick"])

    def props(self) -> dict:
        """GET /props - свойства сервера (default_generation_settings.n_ctx, chat_template ...)."""
        return self._request("GET", "/props", timeout=self.timeouts["quick"])
