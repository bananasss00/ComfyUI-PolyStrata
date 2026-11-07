// js/polystrata_cvec.js — ComfyUI-PolyStrata: нода PolyStrataControlVector.
//
// Зачем: у ноды 19 виджетов, но в каждый момент осмысленна только часть:
//   single_layer   — только при dir_mode = single;
//   layer_start/end — только при layer_range = «вручную»;
//   vector (дропдаун автоскана) — только пока vector_manual пуст (ручной путь
//                                перекрывает дропдаун).
// Прятать — чистая косметика для читабельности: скрытые виджеты ПРОДОЛЖАЮТ
// хранить и сериализовать значения (сдвигов widgets_values нет — порядок
// виджетов не меняется), поэтому на сохранённые воркфлоу это не влияет никак.
//
// Как: app.registerExtension + beforeRegisterNodeDef (официальный хук,
// https://docs.comfy.org/custom-nodes/js/javascript_hooks) оборачивает
// onNodeCreated; виджеты ищутся ПО ИМЕНАМ (они же ключи INPUT_TYPES nodes.py —
// держать синхронно). Переключение видимости — widget.hidden (современные
// фронтенды) + подмена computeSize (старые litegraph, сворачивает в макете);
// всё в try/catch: если что-то не так — просто показываем все виджеты.
//
// Миграций widgets_values здесь НЕТ и не нужно: класс новый (v0.10.0), старых
// воркфлоу с ним не существует. Порядок виджетов заморожен с v0.10.0 — новые
// добавляются только в конец (инвариант 17 AGENTS.md).

import { app } from "../../scripts/app.js";

const NODE_KEY = "PolyStrataControlVector";

// = значения combos из nodes.py — ДЕРЖАТЬ СИНХРОННО.
const DIR_SINGLE = "single";
const LAYERS_MANUAL = "вручную";       // начало значения «вручную (start..end включительно)»

function widget(node, name) {
    return (node.widgets || []).find((w) => w.name === name) || null;
}

function setHidden(node, name, hide) {
    const w = widget(node, name);
    if (!w) return;
    try {
        if (hide) {
            if (!w._psOrigComputeSize) w._psOrigComputeSize = w.computeSize;
            w.computeSize = () => [0, -4];      // старые фронтенды: схлопнуть в макете
            w.hidden = true;                    // новые фронтенды
        } else {
            if (w._psOrigComputeSize) {
                w.computeSize = w._psOrigComputeSize;
                w._psOrigComputeSize = null;
            }
            w.hidden = false;
        }
    } catch (err) {
        console.warn("[PolyStrata] cvec: виджет не спрятан:", name, err);
    }
}

function syncVisibility(node) {
    try {
        const dirMode = String(widget(node, "dir_mode")?.value || "");
        const range = String(widget(node, "layer_range")?.value || "");
        const manual = String(widget(node, "vector_manual")?.value || "").trim();

        setHidden(node, "single_layer", !dirMode.startsWith(DIR_SINGLE));
        setHidden(node, "layer_start", !range.startsWith(LAYERS_MANUAL));
        setHidden(node, "layer_end", !range.startsWith(LAYERS_MANUAL));
        setHidden(node, "vector", manual.length > 0);

        if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true);
    } catch (err) {
        console.warn("[PolyStrata] cvec: видимость виджетов не обновлена:", err);
    }
}

app.registerExtension({
    name: "ComfyUI.PolyStrata.controlVector",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if ((nodeData?.name || nodeType?.ComfyClass) !== NODE_KEY) return;
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated ? origCreated.apply(this, arguments) : undefined;
            try {
                // реакция на смену значений: оборачиваем callback каждого виджета
                (this.widgets || []).forEach((w) => {
                    const orig = w.callback;
                    w.callback = (...a) => {
                        const out = typeof orig === "function" ? orig.apply(w, a) : undefined;
                        syncVisibility(this);
                        return out;
                    };
                });
                syncVisibility(this);
            } catch (err) {
                console.warn("[PolyStrata] cvec: расширение пропущено:", err);
            }
            return r;
        };
    },
});
