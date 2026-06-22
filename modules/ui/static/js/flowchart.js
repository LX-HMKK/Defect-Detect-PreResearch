/**
 * 流程图动画 — stroke-dashoffset 逐段绘制 + 标签淡入
 *
 * 使用 IntersectionObserver 监测 4 张 SVG 是否进入视口，
 * 依次动画：节点边框 → 箭头连线 → 文字标签。
 * 每个 SVG 仅触发一次（observer.unobserve）。
 */
(function() {
    'use strict';

    /**
     * 递归查找元素直到超时（处理 Alpine 延迟渲染场景）
     */
    function waitForSelector(selector, callback, maxRetries) {
        maxRetries = maxRetries || 20;
        var retries = 0;
        function check() {
            var el = document.querySelector(selector);
            if (el) {
                callback();
                return;
            }
            retries++;
            if (retries < maxRetries) {
                setTimeout(check, 250);
            }
        }
        check();
    }

    /**
     * 主初始化：查找所有流程图 SVG 并做预处理（不自动播放）。
     * 实际播放由 algo-carousel.js 在卡片推到最前时触发。
     */
    function initFlowcharts() {
        var svgs = document.querySelectorAll('.flowchart-svg');
        if (svgs.length === 0) return;

        svgs.forEach(function(svg) {
            // 延迟到布局完成后预处理
            if (svg.getBoundingClientRect().width > 0) {
                prepareFlowchart(svg);
            } else {
                var rafId = requestAnimationFrame(function check() {
                    if (svg.getBoundingClientRect().width > 0) {
                        prepareFlowchart(svg);
                    } else {
                        rafId = requestAnimationFrame(check);
                    }
                });
                // 最多等待 3 秒，防止无限轮询
                setTimeout(function() { cancelAnimationFrame(rafId); }, 3000);
            }
        });
    }

    /**
     * 预处理：计算每个节点的周长和每条箭头的路径长度，
     * 设置对应的 stroke-dashoffset 初始值。
     */
    function prepareFlowchart(svg) {
        var nodes = svg.querySelectorAll('.fc-node rect');
        var arrows = svg.querySelectorAll('.fc-arrow');
        var labels = svg.querySelectorAll('.fc-label');

        nodes.forEach(function(rect) {
            var w = parseFloat(rect.getAttribute('width') || '0');
            var h = parseFloat(rect.getAttribute('height') || '0');
            if (w <= 0 || h <= 0) {
                // 回退：从 computed style 读取
                w = rect.getBoundingClientRect().width || 80;
                h = rect.getBoundingClientRect().height || 40;
            }
            var perimeter = 2 * (w + h);
            rect.style.setProperty('--node-perimeter', perimeter.toString());
            rect.style.strokeDasharray = perimeter;
            rect.style.strokeDashoffset = perimeter;
        });

        arrows.forEach(function(arrow) {
            try {
                var length = arrow.getTotalLength();
                if (length > 0) {
                    arrow.style.setProperty('--arrow-length', length.toString());
                    arrow.style.strokeDasharray = length;
                    arrow.style.strokeDashoffset = length;
                }
            } catch (e) {
                // getTotalLength 对某些元素可能不可用；回退估算
                arrow.style.strokeDasharray = '200';
                arrow.style.strokeDashoffset = '200';
            }
        });

        labels.forEach(function(label) {
            label.style.opacity = '0';
        });

        svg.classList.add('is-prepared');
    }

    /**
     * 执行动画：
     *   1) 节点边框逐段绘制（stagger 150ms）
     *   2) 箭头连线依次出现（stagger 200ms，在节点完成后开始）
     *   3) 标签淡入（stagger 100ms）
     */
    function animateFlowchart(svg) {
        if (!svg) return;

        // 若尚未预处理，先准备
        if (!svg.classList.contains('is-prepared')) {
            prepareFlowchart(svg);
        }

        var nodes = svg.querySelectorAll('.fc-node rect');
        var arrows = svg.querySelectorAll('.fc-arrow');
        var labels = svg.querySelectorAll('.fc-label');

        var globalDelay = 0;

        // 阶段 1：节点边框绘制
        nodes.forEach(function(rect, i) {
            var perimeter = parseFloat(
                rect.style.getPropertyValue('--node-perimeter') || '300'
            );
            rect.animate([
                { strokeDashoffset: perimeter },
                { strokeDashoffset: 0 }
            ], {
                duration: 600,
                delay: globalDelay + i * 150,
                easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
                fill: 'forwards'
            });
        });

        globalDelay += Math.max(nodes.length * 150, 150) + 200;

        // 阶段 2：箭头连线
        arrows.forEach(function(arrow, i) {
            var length = parseFloat(
                arrow.style.getPropertyValue('--arrow-length') || '100'
            );
            arrow.animate([
                { strokeDashoffset: length },
                { strokeDashoffset: 0 }
            ], {
                duration: 500,
                delay: globalDelay + i * 200,
                easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
                fill: 'forwards'
            });
        });

        globalDelay += Math.max(arrows.length * 200, 200) + 100;

        // 阶段 3：标签淡入
        labels.forEach(function(label, i) {
            label.animate([
                { opacity: 0 },
                { opacity: 1 }
            ], {
                duration: 300,
                delay: globalDelay + i * 80,
                easing: 'cubic-bezier(0, 0, 0.2, 1)',
                fill: 'forwards'
            });
        });
    }

    // ── 启动入口 ──
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            waitForSelector('.flowchart-svg', initFlowcharts);
        });
    } else {
        // DOM 已就绪，但 Alpine 可能尚未渲染；轮询等待
        waitForSelector('.flowchart-svg', initFlowcharts);
    }

    // 挂载到全局以便 algo-carousel.js 手动触发
    window.initFlowcharts = initFlowcharts;
    window.animateFlowchart = animateFlowchart;
})();
