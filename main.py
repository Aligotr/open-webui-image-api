"""
title: Open WebUI - Function - Image Api
author: Aligotr
description: Интеграция OpenWebUI с внешним провайдером генерации изображений - polza.ai
version: 2025.1.1
license: MIT
"""

import os
import re
import json
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Any, Callable, Awaitable, AsyncGenerator, Tuple
from pydantic import BaseModel, Field

import httpx

# ---------------------------------------
# Общие константы и утилиты
# ---------------------------------------

# Регулярное выражение для извлечения ссылок на изображения из markdown-формата ![alt](url)
MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")

# Лимит количества файлов для одного запроса (общий для URL и Base64)
MAX_FILES_TOTAL = 5


@dataclass
class ModelConfig:
    """
    Описание конфигурации модели генерации изображений внешнего провайдера.

    id:
        Идентификатор модели в API провайдера.
    ui_name:
        Имя модели для отображения в интерфейсе OpenWebUI.
    requires:
        Набор дополнительных параметров для запросов к провайдеру.
    """

    id: str
    ui_name: str
    requires: dict[str, Any] = field(default_factory=dict)


# Регистрация поддерживаемых моделей провайдера
MODELS: dict[str, ModelConfig] = {
    "nano-banana": ModelConfig(
        id="nano-banana",
        ui_name="nano-banana",
    ),
    "gemini-3-pro-image-preview": ModelConfig(
        id="gemini-3-pro-image-preview",
        ui_name="nano-banana-3-pro",
        requires={
            "resolution": "1K",
            "aspect_ratio": "1:1",
        },
    ),
}


# ===========================================
# Ядро интеграции с генерацией изображений
# ===========================================


class Pipe:
    """
    Инкапсуляция логики обработки pipe-запросов генерации изображений
    через внешнего провайдера в OpenWebUI.
    """

    def __init__(self) -> None:
        # Переменные админа
        self.valves = self.Valves()
        # Идентификатор pipe в OpenWebUI
        self.id = "polza-image-generator"
        # Префикс имени pipe в UI
        self.name = "polza/"
        # Ссылка на эмиттер событий OpenWebUI
        self.emitter: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None

    class Valves(BaseModel):
        DEBUG_MODE: bool = Field(default=False)

    def pipes(self) -> list[dict]:
        """
        Формирование списка pipe-конфигураций по моделям провайдера.
        """
        return [{"id": model.id, "name": model.ui_name} for model in MODELS.values()]

    async def pipe(
        self,
        body: dict,
        __user__,
        __event_emitter__: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Обработка одного pipe-запроса OpenWebUI.

        Основные этапы:
        - Извлечение промпта и изображений из истории сообщений;
        - Формирование тела запроса к провайдеру;
        - Делегирование генерации изображения клиенту провайдера;
        - Отправка markdown-ссылки на итоговое изображение.
        """
        self.emitter = __event_emitter__

        try:
            # Получение данных о пользователе
            user_role = __user__["role"]

            # Инициализация провайдера генерации изображений
            image_provider = PolzaImageClient(user_role)

            # Проверка наличия API-ключа внешнего провайдера
            if not image_provider.is_configured():
                error_payload = {"error": "API-ключ провайдера не указан в окружении"}
                yield json.dumps(error_payload, ensure_ascii=False)
                return

            # Получение промпта
            last_user_message, images = self._extract_prompt_and_images(
                body["messages"]
            )

            if not last_user_message and not images:
                raise ValueError(
                    "Не найдено пользовательское сообщение с текстом или изображениями"
                )

            if not last_user_message:
                raise ValueError("Не найдено пользовательское сообщение с текстом")

            if last_user_message and last_user_message.lstrip().startswith("### Task:"):
                return

            # Подготовка конфигурации модели и тела запроса
            model_config = self._resolve_model_config(body["model"])
            payload = self._build_provider_payload(
                model_config, last_user_message, images
            )

            if self.valves.DEBUG_MODE:
                # ===========================================
                # Здесь код для отладки плагина
                # ===========================================
                # yield f"{payload}"
                yield f"Роль пользователя: {user_role}\n\n"
                yield f"<details>\n<summary>payload</summary>\n{payload}\n\n</details>\n\n"
                image_url = "https://pic.rutubelist.ru/video/2024-11-25/8f/5b/8f5bde388b695ad35c3e1d9d83405ad4.jpg"
                yield f"![image]({image_url})\n\n"
                await self._emit_status(
                    f"Debug mode: {self.valves.DEBUG_MODE}", done=True
                )

            else:
                # Делегирование запроса клиенту провайдера
                image_url = await image_provider.generate_image(
                    payload,
                    on_status=self._emit_status,
                )

                # Формирование итогового ответа для OpenWebUI
                if image_url:
                    yield f"![image]({image_url})"
                    await self._emit_status("Изображение сгенерировано", done=True)
                else:
                    await self._emit_status("Ошибка генерации изображения", done=True)

        except (ValueError, RuntimeError, TimeoutError, httpx.HTTPError) as e:
            error_message = f"Ошибка генерации изображения: {e}"
            await self._emit_status(error_message, done=True)

    # ==============================
    # Вспомогательные методы Pipe
    # ==============================

    async def _emit_status(self, description: str, done: bool = False) -> None:
        """
        Отправка статуса выполнения через эмиттер OpenWebUI.
        """
        if not self.emitter:
            return

        await self.emitter(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                },
            }
        )

    def _resolve_model_config(self, raw_model_id: str) -> ModelConfig:
        """
        Разрешение идентификатора модели в конфигурацию из списка MODELS.

        Поддерживаемые форматы идентификатора:
        - "images.nano-banana" -> "nano-banana";
        - "nano-banana" -> "nano-banana".
        """
        model_id = (
            raw_model_id.split(".", 1)[1] if "." in raw_model_id else raw_model_id
        )
        model_config = MODELS.get(model_id)
        if not model_config:
            raise ValueError(f"Модель '{model_id}' не найдена в списке MODELS")
        return model_config

    def _extract_prompt_and_images(
        self,
        messages: list[dict[str, Any]],
    ) -> Tuple[Optional[str], list[str]]:
        """
        Извлечение текста последнего пользовательского сообщения и связанных изображений.

        Поведение:
        - Поиск последнего сообщения с role == "user";
        - Извлечение текста:
            - из строки целиком (обрезка до 30000 символов);
            - из частей с type == "text" (обрезка до 30000 символов);
        - Извлечение ссылок на изображения из user-сообщения (type == "image_url");
        - При отсутствии изображений у пользователя — анализ предыдущего
          assistant-сообщения:
            - извлечение markdown-ссылок ![...](url) из текста;
            - ограничение общим лимитом MAX_FILES_TOTAL.
        """
        text_prompt: Optional[str] = None
        images: list[str] = []

        # Поиск индекса последнего сообщения пользователя
        last_user_idx: Optional[int] = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return None, []

        user_message = messages[last_user_idx]
        content = user_message.get("content")

        # Извлечение текста и изображений из последнего user-сообщения
        if isinstance(content, str):
            text_prompt = content[:30000]
        elif isinstance(content, list):
            text_buffer: list[str] = []
            for part in content:
                part_type = part.get("type")
                if part_type == "text":
                    text_buffer.append(part.get("text") or "")
                elif part_type == "image_url":
                    image_url = part.get("image_url", {}).get("url")
                    if image_url:
                        images.append(image_url)
            if text_buffer:
                text_prompt = "\n".join(text_buffer)[:30000]

        # Извлечение изображений из предыдущего assistant-сообщения при их отсутствии у user
        if not images and last_user_idx - 1 >= 0:
            prev_msg = messages[last_user_idx - 1]
            if prev_msg.get("role") == "assistant":
                a_content = prev_msg.get("content")
                found_urls: list[str] = []

                if isinstance(a_content, str):
                    found_urls = MD_IMAGE_RE.findall(a_content)
                elif isinstance(a_content, list):
                    buf: list[str] = []
                    for part in a_content:
                        if part.get("type") == "text":
                            buf.append(part.get("text") or "")
                    if buf:
                        found_urls = MD_IMAGE_RE.findall("\n".join(buf))

                if found_urls:
                    limit = MAX_FILES_TOTAL - len(images)
                    if limit > 0:
                        images.extend(found_urls[:limit])

        return text_prompt, images

    def _build_files_payload(
        self,
        images: list[str],
    ) -> dict[str, list[str]]:
        """
        Формирование полей filesUrl и filesBase64 для запроса к провайдеру.

        Правила разбора:
        - data:image/...;base64,... -> filesBase64;
        - http/https-ссылки -> filesUrl;
        - суммарное количество элементов (filesUrl + filesBase64) <= MAX_FILES_TOTAL.
        """
        files_url: list[str] = []
        files_b64: list[str] = []

        for img in images:
            if len(files_url) + len(files_b64) >= MAX_FILES_TOTAL:
                break

            if img.startswith("data:image/"):
                files_b64.append(img)
            elif img.startswith(("http://", "https://")):
                files_url.append(img)

        result: dict[str, list[str]] = {}
        if files_url:
            result["filesUrl"] = files_url
        if files_b64:
            result["filesBase64"] = files_b64

        return result

    def _build_provider_payload(
        self,
        model_config: ModelConfig,
        last_user_message: str,
        images: list[str],
    ) -> dict[str, Any]:
        """
        Формирование тела запроса к внешнему провайдеру генерации изображений.

        Поведение:
        - Извлечение последнего пользовательского текста и изображений;
        - Валидация наличия пользовательского текста;
        - Добавление обязательных параметров модели (requires);
        - Преобразование изображений в файлы filesUrl / filesBase64.
        """
        payload: dict[str, Any] = {
            "model": model_config.id,
            "prompt": last_user_message,
        }

        if model_config.requires:
            payload.update(model_config.requires)

        if images:
            files_dict = self._build_files_payload(images)
            payload.update(files_dict)

        return payload


# =================================================
# Клиент провайдера (polza.ai)
# =================================================

POLZAAI_API_BASE_URL = "https://api.polza.ai/api/v1"
POLZAAI_API_KEY_ADMIN = os.getenv("POLZAAI_API_KEY_ADMIN", "")
POLZAAI_API_KEY_USER = os.getenv("POLZAAI_API_KEY_USER", "")


class PolzaImageClient:
    """
    Инкапсуляция взаимодействия с API polza.ai для генерации изображений.

    Основные задачи:
    - Формирование HTTP-запросов;
    - Обработка кодов ответа и тела ошибок;
    - Ожидание завершения генерации изображения по requestId;
    - Извлечение итогового URL изображения из ответа polza.ai.
    """

    def __init__(self, user_role: str) -> None:
        # Установка API-ключа в зависимости от роли пользователя
        if user_role == "admin":
            key = POLZAAI_API_KEY_ADMIN
        else:
            key = POLZAAI_API_KEY_USER
        if key and key != "":
            self.api_key = key

    def is_configured(self) -> bool:
        """
        Проверка наличия сконфигурированного API-ключа.
        """
        return bool(self.api_key)

    def _build_auth_headers(self) -> dict[str, str]:
        """
        Формирование заголовков авторизации для запросов.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def _parse_error_text(self, resp: httpx.Response) -> str:
        """
        Извлечение человекочитаемого текста ошибки из ответа.

        Поведение:
        - Попытка парсинга JSON-ответа;
        - Использование полей message или error.message при наличии;
        - При невозможности парсинга — возврат текстового тела ответа.
        """
        try:
            data = resp.json()
        except ValueError:
            return resp.text

        if isinstance(data, dict):
            if isinstance(data.get("message"), str):
                return data["message"]

            error_field = data.get("error")
            if isinstance(error_field, dict) and "message" in error_field:
                return str(error_field["message"])
            if error_field is not None:
                try:
                    return json.dumps(error_field, ensure_ascii=False)
                except (ValueError, RuntimeError, TimeoutError, httpx.HTTPError) as e:
                    print(f"Ошибка чтения ответа от провайдера:\n{e}")

        return resp.text

    async def _wait_for_request_completion(
        self,
        request_id: str,
        on_status: Optional[Callable[[str, bool], Awaitable[None]]] = None,
    ) -> dict[str, Any]:
        """
        Ожидание завершения задачи генерации в polza.ai по requestId.

        Поведение:
        - Выполнение GET /api/v1/images/{id} с интервалом 5 секунд;
        - Ожидание в течение максимум 120 секунд;
        - Возврат JSON-ответа при статусе COMPLETED;
        - Генерация исключения при статусе FAILED или по таймауту.
        """
        loop = asyncio.get_running_loop()
        end_time = loop.time() + 120
        headers = self._build_auth_headers()

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                resp = await client.get(
                    f"{POLZAAI_API_BASE_URL}/images/{request_id}",
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

                status = data.get("status")
                if on_status:
                    await on_status(
                        f"Ожидание результата от провайдера (status={status})",
                        False,
                    )

                if status == "COMPLETED":
                    return data

                if status == "FAILED":
                    raise RuntimeError(
                        "polza.ai: статус FAILED при генерации изображения"
                    )

                if loop.time() > end_time:
                    raise TimeoutError(
                        "polza.ai: превышено время ожидания генерации изображения"
                    )

                await asyncio.sleep(5.0)

    async def generate_image(
        self,
        payload: dict[str, Any],
        on_status: Optional[Callable[[str, bool], Awaitable[None]]] = None,
    ) -> Optional[str]:
        """
        Создание задачи генерации изображения в polza.ai и получение итогового URL.

        Основные этапы:
        - POST /images/generations — создание задачи:
            - ожидание кода 201;
            - извлечение поля requestId из ответа;
            - обработка ошибок polza.ai через JSON-поля message / error;
        - Ожидание завершения задачи (_wait_for_request_completion) по requestId;
        - Извлечение поля url из ответа со статусом COMPLETED.
        """
        if not self.api_key:
            raise RuntimeError("polza.ai: API-ключ не сконфигурирован")

        headers = {
            **self._build_auth_headers(),
            "Content-Type": "application/json",
        }

        # Создание задачи генерации изображения
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{POLZAAI_API_BASE_URL}/images/generations",
                headers=headers,
                json=payload,
            )

        if resp.status_code != 201:
            error_text = self._parse_error_text(resp)
            msg = f"Ошибка создания задачи polza.ai: {error_text}"
            if on_status:
                await on_status(msg, True)
            return None

        create_data = resp.json()
        request_id = create_data.get("requestId")
        if not request_id:
            msg = f"polza.ai: отсутствует requestId в ответе: {create_data}"
            if on_status:
                await on_status(msg, True)
            return None

        if on_status:
            await on_status(
                f"Ожидание результата от провайдера (requestId={request_id})...",
                False,
            )

        # Ожидание завершения генерации
        status_data = await self._wait_for_request_completion(
            request_id=request_id,
            on_status=on_status,
        )

        image_url = status_data.get("url")
        if not image_url:
            msg = f"polza.ai: статус COMPLETED, но поле url отсутствует: {status_data}"
            if on_status:
                await on_status(msg, True)
            return None

        return image_url
