/**
 * Hero 液态扭曲封面 - grid-distortion 算法（复刻 trae.cn footer）
 *
 * 真实原理（扒自 trae.cn static/js/async/45772.76496a40.js，组件 grid-distortion）：
 *   - 一张 n×n 数据纹理（RGBA32F），每像素 R/G = (x位移, y位移)
 *   - 每帧：所有网格点位移 ×0.9 衰减（残留拖尾的来源）
 *   - 鼠标移动：在鼠标半径内网格点注入 100*strength*vX*距离权重
 *   - shader 仅做：texture(uTexture, uv - 0.02*offset.rg) —— 用位移场偏移采样文字纹理
 *
 * 被扭曲对象：canvas 2D 画的大标题。纹理为不透明（bg 色背景 + text 色文字），
 *   复刻 trae 满铺 PNG 语义 —— CLAMP 边缘 = bg 色 = 画面 bg，位移过大时无缝衔接不露透明。
 *   早期 alpha-only 版本位移 0.02×125=2.5UV 会把文字采样推到透明边缘 → 全 bg，故改不透明。
 *
 * 与早期 fbm 版差异：fbm 版 shader 内算噪声位移，鼠标移走即消失、无拖尾。
 *   本版用 CPU 网格弹簧 + 残留衰减，鼠标划过留下衰减拖影 —— 这才是 trae "拖影特效" 的本质。
 *
 * 参数（复刻 trae 默认）：grid=30, mouse=0.25, strength=0.15, relaxation=0.9, 位移系数=0.02
 *
 * 暴露 window.HeroFluid = { play, pause, destroy }
 * 复刻 cursor-glow.js：IIFE / passive mousemove / RAF /
 *   prefers-reduced-motion 跳过 / visibilitychange 暂停 / scroll 暂停(120ms) /
 *   运行时监听 reduced-motion 变化
 * 可见性复刻 algo-carousel.js：IntersectionObserver root=.snap-container
 *
 * 降级：reduced-motion / 移动端无 hover / WebGL2 不支持 → 不添加 is-fluid-active，
 *   CSS 静态兜底标题(.hero-title--fluid-fallback)显示，canvas 隐藏
 *
 * 性能：视口外暂停 RAF；数据纹理 30×30 仅 900 像素，每帧 texSubImage2D 开销可忽略；
 *       文字纹理不透明，主题切换时重绘（不频繁）；RGBA32F + FLOAT 类型完全匹配零转换
 */
(function () {
    'use strict';

    function noop() {}
    var api = { play: noop, pause: noop, destroy: noop };
    window.HeroFluid = api;

    // ── 无障碍：reduced-motion 完全跳过（不创建 WebGL context）──
    var motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (motionQuery.matches) return;

    // ── 能力检测：移动端 / 无 hover → CSS 静态兜底 ──
    var hoverQuery = window.matchMedia('(hover: hover)');
    var narrowQuery = window.matchMedia('(max-width: 767px)');
    if (!hoverQuery.matches || narrowQuery.matches) return;

    // ── 参数（复刻 trae grid-distortion 默认值，按需微调）──
    var GRID = 30;          // 位移网格分辨率 n×n。值越小网格越粗 → 扭曲块越大越粗犷（像素感越强），
                            //   值越大网格越细 → 扭曲越平滑细腻。trae 默认 15，本项目用 30 偏细腻。
    var MOUSE = 0.25;       // 鼠标影响半径（归一化 0~1，相对画布短边）。鼠标在此半径内的网格点
                            //   才会被注入位移力。值大影响范围广（整片文字被推），值小只扭曲局部。
    var STRENGTH = 0.15;    // 位移注入强度。每帧鼠标速度乘以此值写入网格点。值大扭曲更剧烈（甩一下
                            //   就大幅度变形），值小更克制。配合 100× 倍率后写入数据纹理的 R/G 通道。
    var RELAXATION = 0.9;   // 每帧位移衰减系数（拖尾来源）。每帧所有网格点位移 ×此值。越接近 1 拖尾
                            //   越长（残留久，像墨迹拖痕），越接近 0 瞬间归零（无拖尾）。0.9 ≈ 拖尾
                            //   持续约 0.3~0.5 秒（视刷新率：60Hz 约 0.4s，240Hz 约 0.1s）。
    var DISP_SCALE = 0.02;  // shader 采样位移系数：texture(uTexture, uv - 此值 × offset.rg)。
                            //   把数据纹理的位移值（±百级）映射到 UV 偏移量。值大文字位移幅度大
                            //   （更夸张的拉伸），值小位移微弱。同时写进 fragment shader 源码字符串。

    // ── 状态 ──
    var canvas, gl, snapContainer, layerEl;
    var program, vbo, textTex, dataTex;
    var dataArr;
    var uloc = {};
    var uval = { bgCss: '#0a0a0b', textCss: '#f5f5f7' };  // 文字纹理用 CSS 色
    var rafId = null;
    var isVisible = false;
    var isActive = false;
    var pageVisible = !document.hidden;
    var isScrolling = false, scrollTimer = null;
    var bound = false;

    var T = { x: 0.5, y: 0.5, prevX: 0.5, prevY: 0.5, vX: 0, vY: 0 };
    var heroRect = null;

    // ── 着色器（GLSL ES 3.00，#version 必须首字符）──
    var VS = [
        '#version 300 es',
        'in vec2 aPos;',
        'in vec2 aUv;',
        'out vec2 vUv;',
        'void main(){',
        '    vUv=aUv;',
        '    gl_Position=vec4(aPos,0.0,1.0);',
        '}'
    ].join('\n');

    // 复刻 trae fragment shader：数据纹理 .rg 作为位移，偏移采样文字纹理（不透明，直接用 rgb）
    var FS = [
        '#version 300 es',
        'precision highp float;',
        'uniform sampler2D uTexture;      // 文字（不透明，bg 背景 + text 文字）',
        'uniform sampler2D uDataTexture;  // 网格位移场（RGBA32F）',
        'in vec2 vUv;',
        'out vec4 fragColor;',
        'void main(){',
        '    vec2 uv=vUv;',
        '    vec4 offset=texture(uDataTexture,vUv);',
        '    fragColor=vec4(texture(uTexture, uv - ' + DISP_SCALE + ' * offset.rg).rgb, 1.0);',
        '}'
    ].join('\n');

    function compile(type, src) {
        var sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
            console.error('[HeroFluid] shader 编译失败:', gl.getShaderInfoLog(sh));
            gl.deleteShader(sh);
            return null;
        }
        return sh;
    }

    // ── 读取 CSS 变量 → 文字纹理用色（主题切换时重绘纹理）──
    function readThemeColors() {
        var cs = getComputedStyle(document.documentElement);
        var bg = (cs.getPropertyValue('--bg-root') || '').trim();
        var tx = (cs.getPropertyValue('--text') || '').trim();
        if (bg) uval.bgCss = bg;
        if (tx) uval.textCss = tx;
    }

    // ── 文字纹理：高分辨率抗锯齿大字（保持锐利，被低分辨率位移场扭曲）──
    // 正确分工：文字高分辨率（锐利可读），位移数据纹理低分辨率（粗网格，复刻 trae grid-distortion）。
    //   低分辨率位移 → 文字被"块状"位移场推动 → 自然产生方块化的扭曲形变（这才是"像素感"来源，
    //   来自位移场的粗网格化，而非文字本身降分辨率）。
    // 不透明：CLAMP 边缘 = bg 色 = 画面 bg，位移过大时无缝衔接不露透明（复刻 trae 满铺 PNG 语义）。
    // canvas fillText 不做逐字形字体回退：用 CJK 优先栈，否则中文匹配到无 CJK 字形字体画不出来。
    function createTextTexture() {
        var texW = 2048, texH = 1024;
        var off = document.createElement('canvas');
        off.width = texW; off.height = texH;
        var ctx = off.getContext('2d');

        // bg 背景填充（不透明）
        ctx.fillStyle = uval.bgCss;
        ctx.fillRect(0, 0, texW, texH);

        // text 文字（高分辨率，保留抗锯齿）
        var fontStack = '"PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif';
        ctx.fillStyle = uval.textCss;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        var fontSize = 320;
        ctx.font = '700 ' + fontSize + 'px ' + fontStack;
        var lines = ['无监督', '缺陷检测'];
        var lh = 360;
        var startY = texH / 2 - lh * (lines.length - 1) / 2;
        for (var i = 0; i < lines.length; i++) {
            ctx.fillText(lines[i], texW / 2, startY + i * lh);
        }

        if (textTex) gl.deleteTexture(textTex);
        textTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, textTex);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);   // 不翻转：与 dataTex / y-down 鼠标统一坐标系
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, off);
        // LINEAR：文字采样平滑锐利（高分辨率抗锯齿）
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, textTex);
        gl.uniform1i(uloc.uTexture, 0);
    }

    // ── 数据纹理：n×n RGBA32F（低分辨率粗网格），R/G = (x位移, y位移) ──
    // 正确分工：文字高分辨率锐利，位移场低分辨率（GRID=30，复刻 trae grid-distortion）。
    //   NEAREST 采样：低分辨率位移场不插值，相邻像素位移跳变 → 文字被阶梯状位移推动 →
    //   自然产生方块化的扭曲形变（"像素感"来自位移粗网格化，而非文字本身降分辨率）。
    // 复刻 trae FloatType：RGBA32F + FLOAT + Float32Array 类型完全匹配，零转换。
    function createDataTexture() {
        var n = GRID;
        dataArr = new Float32Array(4 * n * n);
        for (var h = 0; h < n * n; h++) {
            dataArr[4 * h] = 255 * Math.random() - 125;       // R 初始噪声 [-125,130]
            dataArr[4 * h + 1] = 255 * Math.random() - 125;    // G 初始噪声
        }
        // 显式切到 unit1：bindTexture 作用于当前 active unit，若不切会覆盖 unit0 的 textTex
        gl.activeTexture(gl.TEXTURE1);
        dataTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, dataTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, n, n, 0, gl.RGBA, gl.FLOAT, dataArr);
        // NEAREST：保留位移阶梯跳变，产生方块化扭曲（像素感来源）
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }

    function initGL() {
        canvas = document.querySelector('.hero-fluid-canvas');
        layerEl = document.querySelector('.hero-fluid-layer');
        snapContainer = document.querySelector('.snap-container');
        if (!canvas || !layerEl || !snapContainer) return false;

        gl = canvas.getContext('webgl2', {
            antialias: false,
            alpha: false,
            premultipliedAlpha: false,
            powerPreference: 'low-power'
        });
        if (!gl) return false;

        var vs = compile(gl.VERTEX_SHADER, VS);
        var fs = compile(gl.FRAGMENT_SHADER, FS);
        if (!vs || !fs) return false;

        program = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('[HeroFluid] program 链接失败:', gl.getProgramInfoLog(program));
            return false;
        }
        gl.useProgram(program);

        // 全屏 quad：pos + uv（TRIANGLE_STRIP 4 顶点，交错 pos.xy,uv.xy）
        // uv y 翻转：使画面顶=纹理顶（与不 FLIP 的文字纹理 + y-down 鼠标统一坐标系）
        vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1, 0, 1,
            1, -1, 1, 1,
            -1, 1, 0, 0,
            1, 1, 1, 0
        ]), gl.STATIC_DRAW);
        var aPos = gl.getAttribLocation(program, 'aPos');
        var aUv = gl.getAttribLocation(program, 'aUv');
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
        gl.enableVertexAttribArray(aUv);
        gl.vertexAttribPointer(aUv, 2, gl.FLOAT, false, 16, 8);

        ['uTexture', 'uDataTexture'].forEach(function (name) {
            uloc[name] = gl.getUniformLocation(program, name);
        });

        readThemeColors();

        // 文字纹理 → unit 0
        createTextTexture();

        // 数据纹理 → unit 1
        createDataTexture();
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, dataTex);
        gl.uniform1i(uloc.uDataTexture, 1);

        return true;
    }

    function resize() {
        if (!canvas || !gl) return;
        var dpr = Math.min(window.devicePixelRatio || 1, 2);
        var w = Math.max(1, Math.round(window.innerWidth * dpr));
        var h = Math.max(1, Math.round(window.innerHeight * dpr));
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        gl.viewport(0, 0, w, h);
        heroRect = null;
    }

    // ── 鼠标：raw 归一化坐标 + 帧间速度（复刻 trae，无 lerp）──
    function ensureRect() {
        if (!heroRect) heroRect = layerEl.getBoundingClientRect();
        return heroRect;
    }
    function onMove(e) {
        var rect = ensureRect();
        var x = (e.clientX - rect.left) / rect.width;
        var y = (e.clientY - rect.top) / rect.height;   // y-down，与文字纹理 / dataTex 统一
        T.vX = x - T.prevX;
        T.vY = y - T.prevY;
        T.x = x; T.y = y; T.prevX = x; T.prevY = y;
    }
    function onLeave() {
        // 复刻 trae mouseleave：仅重置鼠标状态，数据纹理残留继续自然衰减
        T.x = 0; T.y = 0; T.prevX = 0; T.prevY = 0; T.vX = 0; T.vY = 0;
    }

    // ── 核心：更新数据纹理（衰减 + 鼠标注入），复刻 trae 每帧逻辑 ──
    function updateGrid() {
        var n = GRID;
        var e = dataArr;
        var s, m, v;

        // 1) 全局衰减（拖尾来源）
        for (s = 0; s < n * n; s++) {
            e[4 * s] *= RELAXATION;
            e[4 * s + 1] *= RELAXATION;
        }

        // 2) 鼠标半径内注入力
        var d = n * T.x;
        var u = n * T.y;
        var p = n * MOUSE;
        for (m = 0; m < n; m++) {
            for (v = 0; v < n; v++) {
                var f = (d - m) * (d - m) + (u - v) * (u - v);
                if (f < p * p) {
                    var idx = 4 * (m + n * v);
                    var h = Math.min(p / Math.sqrt(f), 10);
                    e[idx] += 100 * STRENGTH * T.vX * h;
                    e[idx + 1] -= 100 * STRENGTH * T.vY * h;
                }
            }
        }

        // 显式切到 unit1 再 bind/texSubImage，避免覆盖 unit0 的 textTex
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, dataTex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, n, n, gl.RGBA, gl.FLOAT, e);
    }

    function frame() {
        if (!isActive || !isVisible || !pageVisible || isScrolling) {
            rafId = requestAnimationFrame(frame);
            return;
        }
        resize();
        updateGrid();
        // drawArrays 前显式重绑两个 texture unit，确保 uTexture/uDataTexture 采样正确
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, textTex);
        gl.uniform1i(uloc.uTexture, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, dataTex);
        gl.uniform1i(uloc.uDataTexture, 1);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        rafId = requestAnimationFrame(frame);
    }

    function startRAF() { if (rafId == null) rafId = requestAnimationFrame(frame); }
    function stopRAF() { if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; } }

    function setupVisibility() {
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                isVisible = entry.isIntersecting;
                if (isVisible && isActive) startRAF();
                else stopRAF();
            });
        }, { root: snapContainer, threshold: 0.1 });
        var s0 = document.getElementById('s0');
        if (s0) io.observe(s0);
    }

    function bindEvents() {
        if (bound) return;
        bound = true;
        window.addEventListener('mousemove', onMove, { passive: true });
        layerEl.addEventListener('mouseenter', function () { heroRect = null; }, { passive: true });
        layerEl.addEventListener('mouseleave', onLeave, { passive: true });
        window.addEventListener('resize', resize, { passive: true });

        window.addEventListener('scroll', function () {
            isScrolling = true;
            heroRect = null;
            clearTimeout(scrollTimer);
            scrollTimer = setTimeout(function () { isScrolling = false; }, 120);
        }, { passive: true });

        document.addEventListener('visibilitychange', function () {
            pageVisible = !document.hidden;
            if (!pageVisible) stopRAF();
            else if (isVisible && isActive) startRAF();
        });

        // 主题切换：重读颜色 + 重绘文字纹理（不透明纹理含色，需重绘）
        new MutationObserver(function () {
            if (!gl) return;
            readThemeColors();
            createTextTexture();
        }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

        motionQuery.addEventListener('change', function (e) {
            if (e.matches) destroy();
        });
    }

    function play() {
        if (isActive) return;
        if (!gl && !initGL()) { destroy(); return; }
        isActive = true;
        layerEl.classList.add('is-fluid-active');
        bindEvents();
        setupVisibility();
        resize();
        startRAF();
    }

    function pause() { isActive = false; stopRAF(); }

    function destroy() {
        isActive = false;
        stopRAF();
        if (layerEl) layerEl.classList.remove('is-fluid-active');
        if (gl) {
            if (program) gl.deleteProgram(program);
            if (vbo) gl.deleteBuffer(vbo);
            if (textTex) gl.deleteTexture(textTex);
            if (dataTex) gl.deleteTexture(dataTex);
            var lose = gl.getExtension('WEBGL_lose_context');
            if (lose) lose.loseContext();
            gl = null;
        }
        if (canvas) canvas.style.display = 'none';
    }

    api.play = play; api.pause = pause; api.destroy = destroy;

    // ── 启动时机：alpine:initialized + setTimeout 300ms ──
    document.addEventListener('alpine:initialized', function () { setTimeout(play, 300); });
    if (window.Alpine && window.Alpine.version) setTimeout(play, 300);
})();
