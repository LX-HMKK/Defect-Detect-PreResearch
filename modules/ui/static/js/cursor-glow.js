/**
 * 光标光晕增强版 — 滞后追踪 + 强度分区 + CSS 变量
 *
 * 工作原理:
 *   1. 监听 document mousemove（passive），记录目标位置（归一化 0-1）
 *   2. requestAnimationFrame 循环，每帧以 6% 的速率追赶目标（"重感"拖尾）
 *   3. 将当前位置写入 CSS 自定义属性 --cursor-x / --cursor-y（百分比 0-100）
 *   4. mouseover 检测交互元素，设置 --glow-intensity（0.4 基数 / 1.0 增强）
 *   5. CSS 侧通过 body::after 的 radial-gradient 渲染追踪光晕
 *
 * 性能: RAF 天然对齐 ~16ms，mousemove 使用 passive 模式
 */
(function() {
    var targetX = 0.5;
    var targetY = 0.5;
    var currentX = 0.5;
    var currentY = 0.5;

    // ── 鼠标轨迹追踪（不阻塞滚动）──
    document.addEventListener('mousemove', function(e) {
        targetX = e.clientX / window.innerWidth;
        targetY = e.clientY / window.innerHeight;
    }, { passive: true });

    // ── 强度分区：鼠标悬停交互元素时增强光晕 ──
    document.addEventListener('mouseover', function(e) {
        var el = e.target.closest(
            '.algo-card, .flowchart-card, .result-card, .compare-slot, ' +
            '.theme-capsule, .btn-inference, .btn-compare, .upload-zone'
        );
        document.documentElement.style.setProperty(
            '--glow-intensity',
            el ? '1' : '0.4'
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

    // ── RAF 循环：滞后追踪 ──
    function update() {
        // 多级 lerp：每帧向目标移动 6%（重感拖尾）
        currentX += (targetX - currentX) * 0.06;
        currentY += (targetY - currentY) * 0.06;

        // 写入 CSS 变量（百分比 0-100，保留 1 位小数）
        document.documentElement.style.setProperty(
            '--cursor-x',
            (currentX * 100).toFixed(1)
        );
        document.documentElement.style.setProperty(
            '--cursor-y',
            (currentY * 100).toFixed(1)
        );

        requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
})();
