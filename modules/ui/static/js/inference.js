/**
 * Inference runner — SSE 流式推理客户端 + 可视化交互
 *
 * 包含：
 *   1. InferenceRunner — SSE 事件流消费器
 *   2. imageCompare — Alpine 组件：原图/热力图对比滑块
 *   3. setupHeatmapTooltip — 热力图 hover 异常得分 tooltip
 *   4. setupBboxOverlays — bbox 叠加层（hover 高亮）
 */

/* ═══════════════════════════════════════════════════════════════════════════
   1. SSE 推理客户端
   ═══════════════════════════════════════════════════════════════════════════ */
const InferenceRunner = {
    abortController: null,

    /**
     * 发起 SSE 推理请求
     * @param {File} imageFile - 上传的图片文件
     * @param {string} model - 模型 key (patchcore/padim/fre/draem)
     * @param {string} dataset - 数据集名称
     * @param {Object} callbacks
     * @param {Function} callbacks.onProgress - 进度回调 ({stage, message, pct})
     * @param {Function} callbacks.onResult - 结果回调 (resultData)
     * @param {Function} callbacks.onError - 错误回调 (message)
     * @param {Function} callbacks.onDone - 完成回调
     */
    async run(imageFile, model, dataset, callbacks) {
        const { onProgress, onResult, onError, onDone } = callbacks;

        // 取消已有请求
        if (this.abortController) {
            this.abortController.abort();
        }
        this.abortController = new AbortController();

        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('model', model);
        formData.append('dataset', dataset);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData,
                signal: this.abortController.signal,
            });

            if (!response.ok) {
                onError(`HTTP ${response.status}: ${response.statusText}`);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // 归一化行尾：\r\n → \n（sse-starlette 使用 CRLF，但 JS 解析器期望 LF）
                buffer = buffer.replace(/\r\n/g, '\n');

                // SSE 协议：event/data 行对，以 \n\n 分隔
                while (buffer.includes('\n\n')) {
                    const idx = buffer.indexOf('\n\n');
                    const chunk = buffer.slice(0, idx);
                    buffer = buffer.slice(idx + 2);

                    let eventType = '';
                    let dataStr = '';

                    for (const line of chunk.split('\n')) {
                        if (line.startsWith('event: ')) {
                            eventType = line.slice(7).trim();
                        } else if (line.startsWith('data: ')) {
                            dataStr = line.slice(6);
                        }
                    }

                    if (!eventType || !dataStr) continue;

                    try {
                        const data = JSON.parse(dataStr);
                        switch (eventType) {
                            case 'progress':
                                onProgress(data);
                                break;
                            case 'result':
                                onResult(data);
                                break;
                            case 'error':
                                onError(data.message || '未知错误');
                                return;
                            case 'done':
                                onDone();
                                return;
                        }
                    } catch (e) {
                        console.warn('[inference] SSE 数据解析失败:', dataStr);
                    }
                }
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                console.log('[inference] 请求已取消');
                return;
            }
            onError(err.message || '网络错误');
        }
    },

    /** 取消当前推理请求 */
    cancel() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
    }
};


/* ═══════════════════════════════════════════════════════════════════════════
   2. 图片对比滑块 — Alpine 组件
   ═══════════════════════════════════════════════════════════════════════════ */
document.addEventListener('alpine:init', function () {
    Alpine.data('imageCompare', function () {
        return {
            sliderPos: 50,
            dragging: false,

            init: function () {
                var self = this;
                var onMove = function (e) {
                    if (!self.dragging) return;
                    var container = self.$el.querySelector('.compare-container');
                    if (!container) return;
                    var rect = container.getBoundingClientRect();
                    var x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
                    self.sliderPos = Math.max(0, Math.min(100, (x / rect.width) * 100));
                };
                var onUp = function () {
                    self.dragging = false;
                };

                window.addEventListener('mousemove', onMove);
                window.addEventListener('mouseup', onUp);
                window.addEventListener('touchmove', onMove, { passive: true });
                window.addEventListener('touchend', onUp);
            },

            startDrag: function (e) {
                this.dragging = true;
                e.preventDefault();
            }
        };
    });
});


/* ═══════════════════════════════════════════════════════════════════════════
   3. 热力图 Tooltip — hover 显示异常得分
   ═══════════════════════════════════════════════════════════════════════════ */
function setupHeatmapTooltip(anomalyMapEl, heatmapImgEl) {
    if (!anomalyMapEl || !heatmapImgEl) return;

    // 等待图片加载完成
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');

    function initCanvas() {
        canvas.width = anomalyMapEl.naturalWidth;
        canvas.height = anomalyMapEl.naturalHeight;
        ctx.drawImage(anomalyMapEl, 0, 0);
    }

    if (anomalyMapEl.complete) {
        initCanvas();
    } else {
        anomalyMapEl.addEventListener('load', initCanvas, { once: true });
    }

    // 创建 tooltip 元素
    var tooltip = document.createElement('div');
    tooltip.className = 'hm-tooltip';
    tooltip.innerHTML = '<span class="hm-tooltip-label">异常得分</span><span class="hm-tooltip-value"></span>';
    document.body.appendChild(tooltip);

    // 捕获 handler 引用以便后续清除
    var onMove = function (e) {
        if (canvas.width === 0 || canvas.height === 0) return;

        var rect = heatmapImgEl.getBoundingClientRect();
        var scaleX = rect.width / canvas.width;
        var scaleY = rect.height / canvas.height;
        var px = Math.floor((e.clientX - rect.left) / scaleX);
        var py = Math.floor((e.clientY - rect.top) / scaleY);

        if (px < 0 || py < 0 || px >= canvas.width || py >= canvas.height) {
            tooltip.style.display = 'none';
            return;
        }

        var pixel = ctx.getImageData(px, py, 1, 1).data;
        // 灰度图：R 通道即为异常得分 (0-255 → 0.0-1.0)
        var score = pixel[0] / 255;
        tooltip.querySelector('.hm-tooltip-value').textContent = score.toFixed(4);
        tooltip.style.display = 'block';

        // 定位：光标右上方，超出右边界时翻转到左侧
        var tooltipX = e.clientX + 16;
        var tooltipY = e.clientY - 40;
        var tw = tooltip.offsetWidth;
        if (tooltipX + tw > window.innerWidth - 10) {
            tooltipX = e.clientX - tw - 16;
        }
        tooltip.style.left = tooltipX + 'px';
        tooltip.style.top = tooltipY + 'px';
    };

    var onLeave = function () {
        tooltip.style.display = 'none';
    };

    heatmapImgEl.addEventListener('mousemove', onMove);
    heatmapImgEl.addEventListener('mouseleave', onLeave);

    // 返回清理函数
    return function () {
        if (tooltip.parentNode) tooltip.parentNode.removeChild(tooltip);
        heatmapImgEl.removeEventListener('mousemove', onMove);
        heatmapImgEl.removeEventListener('mouseleave', onLeave);
    };
}


/* ═══════════════════════════════════════════════════════════════════════════
   4. Bbox 叠加层
   ═══════════════════════════════════════════════════════════════════════════ */
function setupBboxOverlays(containerEl, bboxes, imgEl) {
    if (!containerEl || !imgEl) return [];

    // 清除已有 overlay
    var existing = containerEl.querySelectorAll('.bbox-overlay');
    existing.forEach(function (el) { el.remove(); });

    if (!bboxes || bboxes.length === 0) return [];

    var cleanups = [];

    function updatePositions() {
        var imgRect = imgEl.getBoundingClientRect();
        var containerRect = containerEl.getBoundingClientRect();

        var naturalW = imgEl.naturalWidth;
        var naturalH = imgEl.naturalHeight;
        var displayW = imgRect.width;
        var displayH = imgRect.height;

        // object-fit: contain 会将图片等比缩放至完全可见，
        // 居中放置在 <img> 元素内。需要计算实际图片内容区域
        // 的缩放比和偏移，而非直接用 <img> 元素尺寸。
        var scale = Math.min(displayW / naturalW, displayH / naturalH);
        var renderedW = naturalW * scale;
        var renderedH = naturalH * scale;

        // 图片内容在 <img> 元素内的居中偏移
        var contentOffsetX = (displayW - renderedW) / 2;
        var contentOffsetY = (displayH - renderedH) / 2;

        // <img> 元素相对于容器的偏移
        var imgOffsetX = imgRect.left - containerRect.left;
        var imgOffsetY = imgRect.top - containerRect.top;

        bboxes.forEach(function (bbox, i) {
            var x = bbox[0], y = bbox[1], w = bbox[2], h = bbox[3], score = bbox[4];
            var overlay = containerEl.querySelector('.bbox-overlay-' + i);
            if (!overlay) return;

            overlay.style.left = (imgOffsetX + contentOffsetX + x * scale) + 'px';
            overlay.style.top = (imgOffsetY + contentOffsetY + y * scale) + 'px';
            overlay.style.width = (w * scale) + 'px';
            overlay.style.height = (h * scale) + 'px';
        });
    }

    // 创建 overlay div
    bboxes.forEach(function (bbox, i) {
        var score = bbox[4];
        var div = document.createElement('div');
        div.className = 'bbox-overlay bbox-overlay-' + i;
        div.title = '缺陷区域 · 得分: ' + score.toFixed(4);
        containerEl.appendChild(div);
    });

    // 初始定位
    updatePositions();

    // 监听尺寸变化
    var resizeObserver = new ResizeObserver(function () {
        updatePositions();
    });
    resizeObserver.observe(imgEl);
    cleanups.push(function () { resizeObserver.disconnect(); });

    // 图片加载完成时重新定位
    var onLoad = function () { updatePositions(); };
    imgEl.addEventListener('load', onLoad);
    cleanups.push(function () { imgEl.removeEventListener('load', onLoad); });

    return cleanups;
}


/* ═══════════════════════════════════════════════════════════════════════════
   5. 通用清理器 — 追踪所有动态创建的监听器/observer
   ═══════════════════════════════════════════════════════════════════════════ */
var _inferenceCleanups = [];

function registerCleanup(fn) {
    if (typeof fn === 'function') {
        _inferenceCleanups.push(fn);
    }
}

function runAllCleanups() {
    _inferenceCleanups.forEach(function (fn) { try { fn(); } catch (e) {} });
    _inferenceCleanups = [];
}
