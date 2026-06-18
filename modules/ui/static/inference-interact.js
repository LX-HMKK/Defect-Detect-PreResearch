/**
 * 推理结果交互增强 — 热力图 hover + NMS bbox 高亮
 *
 * 依赖：_format_result() 返回的 HTML 中隐藏的 #anomaly-map-data（灰度图 base64）
 * 和 #bbox-data（bbox JSON）。通过 MutationObserver 监听 DOM 变化，
 * 在推理结果出现时自动初始化交互层。
 */
(function () {
    'use strict';

    var observer = null;
    var initialized = false;
    var _bboxCleanups = [];  // 事件监听器清理函数列表，防止内存泄漏

    // ── 初始化入口 ──────────────────────────────────────
    function initOnce() {
        if (initialized) return;
        var dataImg = document.getElementById('anomaly-map-data');
        var bboxDiv = document.getElementById('bbox-data');
        if (!dataImg || !bboxDiv) return;

        initialized = true;
        setupHeatmapHover(dataImg);
        setupBboxOverlays(bboxDiv);
    }

    // ── 持续监听（推理结果可能多次更新）─────────────────
    function setupObserver() {
        // 每次推理结果更新时，anomaly-map-data 会重新出现
        observer = new MutationObserver(function () {
            initialized = false;
            observer.disconnect();           // 防止自身 DOM 操作触发递归
            cleanupOverlays();
            setTimeout(initOnce, 200);
            setTimeout(function () {         // 操作完成后重新监听
                observer.observe(document.body, { childList: true, subtree: true });
            }, 300);
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    // ── 热力图 hover: 从离屏 canvas 读灰度值 → tooltip ──
    function setupHeatmapHover(dataImg) {
        var heatmapImg = findHeatmapImage();
        if (!heatmapImg) return;

        // 创建离屏 canvas，绘制灰度图
        var canvas = document.createElement('canvas');
        var offCtx = canvas.getContext('2d');
        canvas.width = dataImg.naturalWidth;
        canvas.height = dataImg.naturalHeight;
        offCtx.drawImage(dataImg, 0, 0);

        // 获取或创建 tooltip
        var tooltip = document.getElementById('heatmap-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'heatmap-tooltip';
            tooltip.className = 'heatmap-tooltip';
            document.body.appendChild(tooltip);
        }

        // 获取 heatmap 图片在页面中的位置与缩放比
        function getHeatmapRect() {
            var rect = heatmapImg.getBoundingClientRect();
            return {
                left: rect.left,
                top: rect.top,
                width: rect.width,
                height: rect.height,
                scaleX: rect.width / canvas.width,
                scaleY: rect.height / canvas.height
            };
        }

        heatmapImg.addEventListener('mousemove', function (e) {
            var hm = getHeatmapRect();
            var px = Math.floor((e.clientX - hm.left) / hm.scaleX);
            var py = Math.floor((e.clientY - hm.top) / hm.scaleY);

            if (px < 0 || py < 0 || px >= canvas.width || py >= canvas.height) {
                tooltip.style.display = 'none';
                return;
            }

            // 读取灰度值 (0-255), 映射回 [0, 1] 异常分数
            var pixel = offCtx.getImageData(px, py, 1, 1).data;
            var gray = pixel[0]; // R channel (grayscale, all channels equal)
            var score = gray / 255.0;

            tooltip.innerHTML =
                '<span class="tt-label">异常得分</span>' +
                '<span class="tt-value">' + score.toFixed(4) + '</span>';
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 16) + 'px';
            tooltip.style.top = (e.clientY - 40) + 'px';
        });

        heatmapImg.addEventListener('mouseleave', function () {
            tooltip.style.display = 'none';
        });
    }

    // ── 查找 Gradio 渲染的热力图 <img> ──────────────────
    function findHeatmapImage() {
        // 热力图在第二个 .image-display 容器中（第一个是原图）
        var displays = document.querySelectorAll('.image-display');
        if (displays.length >= 2) {
            var img = displays[1].querySelector('img');
            if (img) return img;
        }
        // 降级：找所有 Gradio 渲染的大图（排除 SVG 图标和隐藏图）
        var allImgs = document.querySelectorAll('img');
        var gradioImgs = [];
        allImgs.forEach(function (img) {
            if (img.src.includes('gradio_api/file=') && img.width > 100) {
                gradioImgs.push(img);
            }
        });
        // 原图是第一个，热力图是第二个
        if (gradioImgs.length >= 3) return gradioImgs[2];
        if (gradioImgs.length >= 2) return gradioImgs[1];
        return null;
    }

    // ── bbox overlay: 创建绝对定位透明 div，hover 高亮 ──
    function setupBboxOverlays(bboxDiv) {
        var bboxes;
        try {
            bboxes = JSON.parse(bboxDiv.getAttribute('data-bboxes') || '[]');
        } catch (e) {
            return;
        }
        if (!bboxes || bboxes.length === 0) return;

        var heatmapImg = findHeatmapImage();
        if (!heatmapImg) return;

        // 找到热力图的父容器作为 overlay 定位参考
        var container = heatmapImg.closest('.svelte-1plpy97') ||
                        heatmapImg.closest('.image-display') ||
                        heatmapImg.parentElement;
        if (!container) return;

        // 确保父容器是定位上下文
        var containerPos = getComputedStyle(container).position;
        if (containerPos === 'static') {
            container.style.position = 'relative';
        }

        // 清除旧 overlay
        container.querySelectorAll('.bbox-overlay').forEach(function (el) { el.remove(); });

        bboxes.forEach(function (bbox, i) {
            var x = bbox[0], y = bbox[1], w = bbox[2], h = bbox[3], score = bbox[4];

            var overlay = document.createElement('div');
            overlay.className = 'bbox-overlay';
            overlay.setAttribute('data-bbox-index', i);
            overlay.setAttribute('title', '缺陷区域 #' + (i + 1) + ' | 得分: ' + score.toFixed(4));
            overlay.style.cssText =
                'position:absolute;' +
                'border:2px solid transparent;' +
                'border-radius:4px;' +
                'cursor:pointer;' +
                'transition:border-color 180ms var(--ease-out, ease),' +
                           'box-shadow 180ms var(--ease-out, ease);' +
                'z-index:10;';

            container.appendChild(overlay);

            // 延迟更新位置（等图片加载完成）
            updateOverlayPosition(overlay, heatmapImg, x, y, w, h);

            // 图片加载或窗口 resize 时更新位置（保存引用以便清理）
            var onLoad = function () {
                updateOverlayPosition(overlay, heatmapImg, x, y, w, h);
            };
            var onResize = function () {
                updateOverlayPosition(overlay, heatmapImg, x, y, w, h);
            };
            heatmapImg.addEventListener('load', onLoad);
            window.addEventListener('resize', onResize);
            _bboxCleanups.push(function () {
                heatmapImg.removeEventListener('load', onLoad);
                window.removeEventListener('resize', onResize);
            });
        });
    }

    function updateOverlayPosition(overlay, img, x, y, w, h) {
        var imgRect = img.getBoundingClientRect();
        var containerRect = overlay.parentElement.getBoundingClientRect();
        var scaleX = imgRect.width / img.naturalWidth;
        var scaleY = imgRect.height / img.naturalHeight;
        var left = imgRect.left - containerRect.left + x * scaleX;
        var top = imgRect.top - containerRect.top + y * scaleY;
        overlay.style.left = left + 'px';
        overlay.style.top = top + 'px';
        overlay.style.width = (w * scaleX) + 'px';
        overlay.style.height = (h * scaleY) + 'px';
    }

    function cleanupOverlays() {
        // 先移除事件监听器（防止内存泄漏）
        _bboxCleanups.forEach(function (fn) { fn(); });
        _bboxCleanups = [];
        // 再移除 DOM 元素
        document.querySelectorAll('.bbox-overlay').forEach(function (el) { el.remove(); });
    }

    // ── 启动 ────────────────────────────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            initOnce();
            setupObserver();
        });
    } else {
        initOnce();
        setupObserver();
    }
})();
