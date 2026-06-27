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
            // 注意：stroke-dasharray / stroke-dashoffset 是 <length> 型属性，
            // 非零值必须带单位（否则整条声明无效，回退到初始值 0，节点会一直可见）。
            var perimeter = 2 * (w + h);
            rect.style.setProperty('--node-perimeter', perimeter + 'px');
            rect.style.strokeDasharray = perimeter + 'px';
            rect.style.strokeDashoffset = perimeter + 'px';
        });

        arrows.forEach(function(arrow) {
            try {
                var length = arrow.getTotalLength();
                if (length > 0) {
                    arrow.style.setProperty('--arrow-length', length + 'px');
                    arrow.style.strokeDasharray = length + 'px';
                    arrow.style.strokeDashoffset = length + 'px';
                }
            } catch (e) {
                // getTotalLength 对某些元素可能不可用；回退估算
                arrow.style.strokeDasharray = '200px';
                arrow.style.strokeDashoffset = '200px';
            }
        });

        labels.forEach(function(label) {
            label.style.opacity = '0';
        });

        // 初始态：SVG 整体隐藏，等待推到最前时 animateFlowchart 展开
        svg.style.visibility = 'hidden';

        svg.classList.add('is-prepared');
    }

    /**
     * 重置流程图到未渲染状态：取消内部所有 WAAPI 动画，并将 SVG 整体隐藏。
     *
     * 用途：卡片退出最前时"关闭渲染"，以便再次推到最前时能从头播放。
     * 与 animateFlowchart 配对使用：resetFlowchart → animateFlowchart。
     *
     * 实现说明：stroke-dashoffset 在当前 Chrome/SVG 渲染栈中无法可靠复位
     * （存在未知高优先级规则强制其回 0），因此主生命周期改用 SVG 整体
     * visibility + scale 驱动：退出 → visibility hidden（关闭）；进入 → visible + 弹性展开（渲染）。
     */
    function resetFlowchart(svg) {
        if (!svg) return;

        // 取消 SVG 自身及内部所有元素的 WAAPI 动画，解除 fill:forwards 对属性的占用
        [svg].concat(Array.from(svg.querySelectorAll('.fc-node rect, .fc-arrow, .fc-label'))).forEach(function (el) {
            if (el.getAnimations) {
                el.getAnimations().forEach(function (a) { a.cancel(); });
            }
        });

        // 关闭渲染：整个 SVG 隐藏（保留布局，避免与 opacity 动画/规则发生 specificity 冲突）
        svg.style.visibility = 'hidden';
    }

    /**
     * 执行动画：
     *   1) SVG 整体弹性展开（opacity + scale）
     *   2) 内部标签 stagger 淡入
     *
     * 支持反复触发：内部先 reset（隐藏+取消动画），再从头展开。
     */
    function animateFlowchart(svg) {
        if (!svg) return;

        if (!svg.classList.contains('is-prepared')) {
            prepareFlowchart(svg);
        }

        // 确保从头开始：先关闭再展开
        resetFlowchart(svg);

        var nodes = svg.querySelectorAll('.fc-node rect');
        var arrows = svg.querySelectorAll('.fc-arrow');
        var labels = svg.querySelectorAll('.fc-label');

        // ── 阶段 0：SVG 整体弹性展开 ──
        svg.style.visibility = 'visible';
        svg.animate([
            { transform: 'scale(0.96)' },
            { transform: 'scale(1)' }
        ], {
            duration: 450,
            easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
            fill: 'forwards'
        });

        var globalDelay = 180; // 等 SVG 展开一小段后再启动内部元素

        // ── 阶段 1：节点边框绘制（dashoffset 动画作为装饰层，若渲染栈支持则可见）──
        nodes.forEach(function(rect, i) {
            var perimeter = parseFloat(
                rect.style.getPropertyValue('--node-perimeter') || '300'
            );
            rect.animate([
                { strokeDashoffset: perimeter + 'px' },
                { strokeDashoffset: '0px' }
            ], {
                duration: 600,
                delay: globalDelay + i * 150,
                easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
                fill: 'forwards'
            });
        });

        globalDelay += Math.max(nodes.length * 150, 150) + 200;

        // ── 阶段 2：箭头连线 ──
        arrows.forEach(function(arrow, i) {
            var length = parseFloat(
                arrow.style.getPropertyValue('--arrow-length') || '100'
            );
            arrow.animate([
                { strokeDashoffset: length + 'px' },
                { strokeDashoffset: '0px' }
            ], {
                duration: 500,
                delay: globalDelay + i * 200,
                easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
                fill: 'forwards'
            });
        });

        globalDelay += Math.max(arrows.length * 200, 200) + 100;

        // ── 阶段 3：标签淡入 ──
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
    window.resetFlowchart = resetFlowchart;
})();
