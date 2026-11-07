// js/polystrata_migration.js — ComfyUI-PolyStrata: МИГРАЦИЯ СТАРЫХ ВОРКФЛОУ.
//
// ЗАЧЕМ. JSON воркфлоу ComfyUI хранит значения виджетов ноды ПОЗИЦИОННО
// (nodes[].widgets_values — массив строго по порядку виджетов), а не по именам.
// Вставка или удаление виджета в новой версии нодпака сдвигает значения всех
// виджетов НИЖЕ: в v0.9.0 в PolyStrataChat вторым после server вставлен дропдаун
// model — без миграции старый промпт «уезжал» в model, пресет — в промпт и т.д.
//
// КАК. При загрузке воркфлоу ComfyUI сначала строит ноду по АКТУАЛЬНОМУ
// определению (виджеты уже созданы), потом вызывает configure(info), где
// info.widgets_values позиционно раскладывается по виджетам. Здесь
// оборачивается nodeType.prototype.configure (хук beforeRegisterNodeDef —
// официально документированный способ менять поведение нод:
// https://docs.comfy.org/custom-nodes/js/javascript_hooks) и ДО вызова
// оригинала значения переписываются под текущий формат.
//
// ДЕТЕКЦИЯ — по «якорю», а не по длине массива. reasoning — всегда одна из
// фиксированных строк, и сразу за ней идёт число (max_tokens); в голове ноды
// такая пара единственная (preset и text — строки, за ними не число):
//   v0.8.x: server, prompt, preset, text, [reasoning, число] -> reasoning на 4
//   v0.9.x: server, model, prompt, preset, text, [reasoning, число] -> на 5
// Поэтому не важно, как фронтенд сериализует seed и control_after_generate
// (плоско, вложенно, отдельным виджетом): хвост после reasoning переносится
// как есть, длина массива не сверяется вовсе.
//
// ВСТРОЕННЫЕ МЕХАНИЗМЫ COMFYUI (почему миграция всё же здесь):
//  - настройка «Restore widget values by name» (Comfy.Workflow.NamedValuesRestore)
//    восстанавливает значения по ИМЕНАМ виджетов и лечит сдвиги глобально, но
//    выключена по умолчанию и зависит от пользователя;
//  - NodeReplace (comfy_api, GET /node_replacements) мигрирует только при
//    ПЕРЕИМЕНОВАНИИ класса ноды (old_node_id -> new_node_id), внутри одного
//    класса значения виджетов не переносит.
// Правило проекта на будущее (см. AGENTS.md): новые виджеты добавляются В КОНЕЦ
// списка входов — тогда старые воркфлоу грузятся и без миграции. Вставка/
// удаление/переименование в середине — только вместе с записью в MIGRATIONS.

import { app } from "../../scripts/app.js";

// = MODEL_SWITCH_OFF из nodes.py — ДЕРЖАТЬ СИНХРОННО (проверяет тест пакета).
const MODEL_SENTINEL = "как на сервере";

const REASONING_OPTIONS = ["default", "none", "low", "medium", "high"];

/** Индекс «якоря»: строка из опций reasoning, за которой сразу число (max_tokens). */
function reasoningAnchor(values) {
    for (let i = 1; i < values.length - 1; i++) {
        if (typeof values[i] === "string" && REASONING_OPTIONS.includes(values[i])
            && typeof values[i + 1] === "number") {
            return i;
        }
    }
    return -1;
}

/**
 * Таблица миграций: name — человеческое описание; detect(values) — узнаёт СВОЙ
 * старый формат; apply(values) — возвращает массив под АКТУАЛЬНЫЙ макет;
 * objectKey/objectValue — что дописать, если фронтенд сохранил значения
 * объектом {имя_виджета: значение} (формат «restore by name» — там сдвигов
 * нет, нужен только дефолт новому виджету). Миграции применяются цепочкой,
 * пока какая-то узнаёт свой формат (v0.7 -> v0.8 -> v0.9 за одну загрузку).
 *
 * НОВАЯ МИГРАЦИЯ ПРИ СМЕНЕ МАКЕТА ВИЖЕТОВ — запись сюда и только сюда.
 */
const MIGRATIONS = {
    PolyStrataChat: [
        {
            name: "v0.8.x -> v0.9.0: виджет model вставлен вторым, после server",
            detect: (values) => Array.isArray(values) && reasoningAnchor(values) === 4,
            apply: (values) => [values[0], MODEL_SENTINEL, ...values.slice(1)],
            objectKey: "model",
            objectValue: MODEL_SENTINEL,
        },
    ],
    // PolyStrataSampling / Media / Memory / Unload: макет виджетов с v0.8.0
    // не менялся — записей нет; появятся при будущих изменениях.
};

function note(name) {
    console.info(`[PolyStrata] воркфлоу мигрирован: ${name}`);
    try {
        app.extensionManager?.toast?.add?.({
            summary: "PolyStrata: старый воркфлоу обновлён",
            description: name,
            severity: "info",
        });
    } catch (_) {
        // тост недоступен (старый фронтенд) — уже залогировали в консоль
    }
}

/** Переписать info.widgets_values под текущий формат (in-place по info). */
function migrateNodeInfo(info, migrations) {
    const values = info?.widgets_values;
    if (values == null) return;

    // Объектная форма значений: позиционного сдвига нет, только дописываем
    // отсутствующие новые виджеты, чтобы не терять дефолт.
    if (!Array.isArray(values)) {
        if (typeof values === "object") {
            for (const m of migrations) {
                if (m.objectKey && !(m.objectKey in values)) {
                    values[m.objectKey] = m.objectValue;
                    note(m.name);
                }
            }
        }
        return;
    }

    // Массив: применяем миграции цепочкой, пока какая-то узнаёт свой формат.
    for (let guard = 0; guard < migrations.length; guard++) {
        const m = migrations.find((x) => x.detect(info.widgets_values));
        if (!m) break;
        info.widgets_values = m.apply(info.widgets_values);
        note(m.name);
    }
}

app.registerExtension({
    name: "ComfyUI.PolyStrata.workflowMigration",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        const nodeKey = nodeData?.name || nodeType?.ComfyClass;
        const migrations = MIGRATIONS[nodeKey];
        if (!migrations || migrations.length === 0) return;
        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            try {
                migrateNodeInfo(info, migrations);
            } catch (err) {
                console.warn("[PolyStrata] миграция воркфлоу пропущена:", err);
            }
            return origConfigure ? origConfigure.call(this, info) : undefined;
        };
    },
});

// Внутренности для тестов (scripts/ в песочнице разработки; на работу
// расширения не влияют: ComfyUI просто импортирует модуль).
export const _test = {
    MODEL_SENTINEL, REASONING_OPTIONS, reasoningAnchor, MIGRATIONS, migrateNodeInfo,
};
