/**
 * 光标光晕 — AirPods Pro 式克制环境细节
 *
 * 工作原理:
 *   1. 监听 document mousemove（passive），记录目标位置（像素）
 *   2. requestAnimationFrame 循环，每帧以 8% 速率 lerp 追赶目标
 *   3. 将当前位置写入 CSS 自定义属性 --cursor-x / --cursor-y（像素）
 *   4. mouseover 检测交互元素，设置 --glow-intensity（0.4 基数 / 1.2 增强）
 *   5. CSS 侧通过 body::after 的 radial-gradient 渲染追踪光晕
 *
 * 无障碍：prefers-reduced-motion 时完全禁用，不注册任何监听器。
 *
 * 性能: RAF 天然对齐 ~16ms，mousemove 使用 passive 模式
 */
(function() {
    'use strict';

    // 无障碍：用户偏好减少动效时，完全跳过光标光晕
    var motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (motionQuery.matches) return;

    var rafId = null;
    var targetX = window.innerWidth / 2;
    var targetY = window.innerHeight / 2;
    var currentX = targetX;
    var currentY = targetY;

    // ── 鼠标轨迹追踪（不阻塞滚动）──
    document.addEventListener('mousemove', function(e) {
        targetX = e.clientX;
        targetY = e.clientY;
    }, { passive: true });

    // ── 强度分区：鼠标悬停交互元素时增强光晕 ──
    // 使用 data-glow-enhance 属性标记需增强光晕的元素
    document.addEventListener('mouseover', function(e) {
        var el = e.target.closest(
            '.algo-card, .flowchart-card, .result-card, .compare-slot, ' +
            '.theme-capsule, .btn-inference, .btn-compare, .upload-zone, ' +
            '.btn-primary, .btn-secondary, .custom-select-trigger'
        );
        document.documentElement.style.setProperty(
            '--glow-intensity',
            el ? '1.2' : '0.4'
        );
    });

    // 鼠标离开文档区域时回到基线强度
    document.addEventListener('mouseleave', function() {
        document.documentElement.style.setProperty('--glow-intensity', '0');
    });

    // 鼠标重新进入时恢复
    document.addEventListener('mouseenter', function() {
        document.documentElement.style.setProperty('--glow-intensity', '0.4');
    });

    // ── RAF 循环：平滑滞后追踪 ──
    function update() {
        // lerp 系数 0.08，让移动更柔和从容
        currentX += (targetX - currentX) * 0.08;
        currentY += (targetY - currentY) * 0.08;

        // 写入 CSS 变量（像素，供 body::after 的 left/top 使用）
        document.documentElement.style.setProperty(
            '--cursor-x',
            currentX.toFixed(1)
        );
        document.documentElement.style.setProperty(
            '--cursor-y',
            currentY.toFixed(1)
        );

        rafId = requestAnimationFrame(update);
    }

    rafId = requestAnimationFrame(update);

    // 窗口大小变化时保持中心点合理
    window.addEventListener('resize', function() {
        targetX = Math.min(targetX, window.innerWidth);
        targetY = Math.min(targetY, window.innerHeight);
    });

    // ── 运行时监听 reduced-motion 变化 ──
    motionQuery.addEventListener('change', function(e) {
        if (e.matches) {
            // 用户开启 reduced-motion：停止 RAF，熄灭光晕
            if (rafId) {
                cancelAnimationFrame(rafId);
                rafId = null;
            }
            document.documentElement.style.setProperty('--glow-intensity', '0');
            document.documentElement.style.setProperty('--cursor-x', (window.innerWidth / 2).toFixed(1));
            document.documentElement.style.setProperty('--cursor-y', (window.innerHeight / 2).toFixed(1));
        }
    });
})();
