/**
 * Alpine.js 全局状态 — App
 *
 * 管理主题切换、导航滚动、数据集/模型列表、推理状态等。
 * 必须在 Alpine.js CDN 之前加载（通过 alpine:init 事件注册）。
 */
document.addEventListener('alpine:init', function () {
    Alpine.data('app', function () {
        return {
            // ─────────────────────────────────────────────
            // 主题
            // ─────────────────────────────────────────────
            theme: 'dark',

            get isDark() {
                return this.theme === 'dark';
            },
            get isLight() {
                return this.theme === 'light';
            },

            // ─────────────────────────────────────────────
            // 健康检查 & Toast 通知
            // ─────────────────────────────────────────────
            healthOk: true,
            healthCheckInterval: null,
            toasts: [],

            init: function () {
                var self = this;

                // 读取保存的主题偏好
                var savedTheme = localStorage.getItem('theme');
                if (savedTheme && (savedTheme === 'dark' || savedTheme === 'light')) {
                    self.theme = savedTheme;
                } else {
                    self.theme = window.matchMedia('(prefers-color-scheme: dark)').matches
                        ? 'dark'
                        : 'light';
                }
                self.applyTheme();

                // 监听系统主题变化（仅当无手动偏好时）
                window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function (e) {
                    if (!localStorage.getItem('theme')) {
                        self.theme = e.matches ? 'dark' : 'light';
                        self.applyTheme();
                    }
                });

                // 获取模型列表
                self.fetchModels();

                // 设置滚动监听（用于导航点更新）
                self.setupScrollObserver();

                // 启动健康检查轮询
                self.startHealthCheck();

                // 设置全局异常处理
                self.setupErrorHandling();

                // 隐藏页面加载遮罩
                setTimeout(function () {
                    var loader = self.$refs.pageLoader;
                    if (loader) {
                        loader.style.pointerEvents = 'none';
                    }
                }, 1500);
            },

            toggleTheme: function () {
                this.theme = this.isDark ? 'light' : 'dark';
                localStorage.setItem('theme', this.theme);
                this.applyTheme();

                // 主题切换时闪光反馈
                var capsule = this.$el.querySelector('.theme-capsule');
                if (capsule) {
                    capsule.style.boxShadow = this.isLight
                        ? '0 0 16px rgba(255, 159, 10, 0.4)'
                        : '0 0 16px rgba(41, 151, 255, 0.4)';
                    setTimeout(function () {
                        capsule.style.boxShadow = '';
                    }, 400);
                }
            },

            applyTheme: function () {
                document.documentElement.setAttribute('data-theme', this.theme);

                // 更新 favicon
                var favicon = document.getElementById('favicon');
                if (favicon) {
                    var darkFav = 'data:image/svg+xml,' + encodeURIComponent(
                        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">' +
                        '<circle cx="16" cy="16" r="16" fill="#1c1c1e"/>' +
                        '<polygon points="16,4 28,16 16,28 4,16" fill="#2997ff" opacity="0.9"/>' +
                        '</svg>'
                    );
                    var lightFav = 'data:image/svg+xml,' + encodeURIComponent(
                        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">' +
                        '<circle cx="16" cy="16" r="16" fill="#e8e8ed"/>' +
                        '<polygon points="16,4 28,16 16,28 4,16" fill="#2997ff" opacity="0.9"/>' +
                        '</svg>'
                    );
                    favicon.href = this.isLight ? lightFav : darkFav;
                }
            },

            /** 健康检查轮询（每 30s） */
            startHealthCheck: function () {
                var self = this;
                this.healthCheckInterval = setInterval(async function () {
                    try {
                        var res = await fetch('/api/health', { signal: AbortSignal.timeout(5000) });
                        var wasOffline = !self.healthOk;
                        self.healthOk = res.ok;
                        if (wasOffline && self.healthOk) {
                            self.showToast('服务已恢复', 'ok');
                        }
                    } catch (e) {
                        if (self.healthOk) {
                            self.healthOk = false;
                            self.showToast('服务离线，正在重连…', 'error');
                        }
                    }
                }, 30000);
            },

            /** Toast 通知系统 */
            showToast: function (message, type) {
                type = type || 'info';
                var id = Date.now();
                this.toasts.push({ id: id, message: message, type: type });
                var self = this;
                setTimeout(function () {
                    self.toasts = self.toasts.filter(function (t) { return t.id !== id; });
                }, 4000);
            },

            /** 全局网络异常处理 */
            setupErrorHandling: function () {
                var self = this;
                window.addEventListener('unhandledrejection', function (event) {
                    console.error('Unhandled rejection:', event.reason);
                    self.showToast('网络请求失败，请检查连接', 'error');
                });

                window.addEventListener('error', function (event) {
                    if (event.target && event.target.tagName === 'IMG') {
                        console.warn('Image load error:', event.target.src);
                    }
                });
            },

            // ─────────────────────────────────────────────
            // 导航
            // ─────────────────────────────────────────────
            scrolled: false,
            currentSection: 0,

            /** 导航点 tooltip 标签 */
            sectionNames: ['算法介绍', '单模型推理', '四模型对比'],

            setupScrollObserver: function () {
                var self = this;

                // 导航栏滚动加深
                window.addEventListener('scroll', function () {
                    self.scrolled = window.scrollY > 100;
                }, { passive: true });

                // IntersectionObserver 检测当前 section（取可见比例最大的）
                var sections = [];
                if (self.$refs.section0) sections.push(self.$refs.section0);
                if (self.$refs.section1) sections.push(self.$refs.section1);
                if (self.$refs.section2) sections.push(self.$refs.section2);

                if (sections.length === 0) return;

                var observer = new IntersectionObserver(
                    function (entries) {
                        var maxRatio = 0;
                        var maxIdx = self.currentSection;

                        entries.forEach(function (entry) {
                            if (entry.intersectionRatio > maxRatio) {
                                maxRatio = entry.intersectionRatio;
                                var idx = sections.indexOf(entry.target);
                                if (idx >= 0) maxIdx = idx;
                            }
                        });

                        if (maxRatio > 0) {
                            self.currentSection = maxIdx;
                        }
                    },
                    { threshold: [0, 0.25, 0.5, 0.75] }
                );

                sections.forEach(function (s) {
                    observer.observe(s);
                });

                // ── 键盘导航（↑↓ 切换 section）──
                window.addEventListener('keydown', function (e) {
                    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                        // 不在 input/select 中时才处理
                        var tag = document.activeElement ? document.activeElement.tagName : '';
                        if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

                        e.preventDefault();
                        var dir = e.key === 'ArrowDown' ? 1 : -1;
                        var next = Math.max(0, Math.min(sections.length - 1, self.currentSection + dir));
                        self.currentSection = next;
                        self.scrollToSection(next);
                    }
                });
            },

            scrollToSection: function (idx) {
                var sections = [];
                if (this.$refs.section0) sections.push(this.$refs.section0);
                if (this.$refs.section1) sections.push(this.$refs.section1);
                if (this.$refs.section2) sections.push(this.$refs.section2);

                var target = sections[idx];
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            },

            // ─────────────────────────────────────────────
            // 鼠标移动（委托给 cursor-glow.js）
            // ─────────────────────────────────────────────
            onMouseMove: function (e) {
                // cursor-glow.js 通过独立 document 监听处理
            },

            // ─────────────────────────────────────────────
            // 数据集 & 模型
            // ─────────────────────────────────────────────
            datasets: [],
            selectedDataset: 'bottle',
            models: [],
            selectedModel: 'patchcore',

            fetchModels: async function () {
                try {
                    var res = await fetch('/api/models');
                    var data = await res.json();
                    this.models = data.models || [];
                    this.datasets = data.datasets || [];
                    if (this.datasets.length > 0) {
                        this.selectedDataset = this.datasets[0];
                    }
                } catch (e) {
                    console.warn('[app] 获取模型列表失败:', e);
                }
            },

            // ─────────────────────────────────────────────
            // 推理状态（P1 实现）
            // ─────────────────────────────────────────────
            inferenceState: 'idle',  // idle | uploaded | loading | inferring | done | error
            inferenceProgress: { stage: '', message: '', pct: 0 },
            resultData: null,
            uploadedFile: null,
            uploadPreviewUrl: null,
            errorMessage: '',

            /** 文件选择回调 */
            onFileSelected: function (event) {
                var file = event.target.files[0];
                if (!file) return;
                if (!file.type.startsWith('image/')) {
                    alert('请上传图片文件 (PNG/JPG/BMP)');
                    return;
                }
                this.uploadedFile = file;
                if (this.uploadPreviewUrl) URL.revokeObjectURL(this.uploadPreviewUrl);
                this.uploadPreviewUrl = URL.createObjectURL(file);
                this.inferenceState = 'uploaded';
                this.resultData = null;
                this.errorMessage = '';
                // 共享图片给四模型对比组件
                window._compareImageFile = file;
                window._appDataset = this.selectedDataset;
            },

            /** 拖拽放置回调 */
            onDrop: function (event) {
                event.preventDefault();
                var file = event.dataTransfer.files[0];
                if (!file) return;
                if (!file.type.startsWith('image/')) {
                    alert('请上传图片文件 (PNG/JPG/BMP)');
                    return;
                }
                this.uploadedFile = file;
                if (this.uploadPreviewUrl) URL.revokeObjectURL(this.uploadPreviewUrl);
                this.uploadPreviewUrl = URL.createObjectURL(file);
                this.inferenceState = 'uploaded';
                this.resultData = null;
                this.errorMessage = '';
                // 共享图片给四模型对比组件
                window._compareImageFile = file;
                window._appDataset = this.selectedDataset;
            },

            /** 开始推理 */
            startInference: function () {
                var self = this;
                if (!self.uploadedFile) return;
                if (self.inferenceState === 'loading' || self.inferenceState === 'inferring') return;

                // 清理上一次的可视化元素
                runAllCleanups();

                self.inferenceState = 'loading';
                self.inferenceProgress = { stage: 'loading_model', message: '正在加载模型...', pct: 10 };

                InferenceRunner.run(self.uploadedFile, self.selectedModel, self.selectedDataset, {
                    onProgress: function (data) {
                        self.inferenceState = 'inferring';
                        self.inferenceProgress = data;
                    },
                    onResult: function (data) {
                        self.resultData = data;
                        self.inferenceState = 'done';
                        // 下一帧触发数字滚动动画 + 可视化交互 + 滚动淡入
                        self.$nextTick(function () {
                            self.animateNumbers();
                            self.setupVisualInteractions();
                            // 重新触发 scroll-reveal 以捕获新出现的结果面板元素
                            setTimeout(function () {
                                if (window.initAllAnimations) window.initAllAnimations();
                            }, 100);
                        });
                    },
                    onError: function (message) {
                        self.inferenceState = 'error';
                        self.errorMessage = message;
                    },
                    onDone: function () {
                        // 结果已通过 onResult 处理
                    }
                });
            },

            /** 数字滚动动画 */
            animateNumbers: function () {
                if (!this.resultData) return;
                var scoreEl = document.querySelector('.result-score-value');
                if (scoreEl) {
                    Anim.numberRoll(scoreEl, 0, this.resultData.score, 600);
                }
                var confEl = document.querySelector('.result-confidence-value');
                if (confEl) {
                    Anim.numberRoll(confEl, 0, this.resultData.confidence * 100, 600, {
                        format: function (v) { return v.toFixed(1) + '%'; }
                    });
                }
            },

            /** 设置可视化交互（tooltip + bbox overlay） */
            setupVisualInteractions: function () {
                var self = this;
                var anomalyEl = self.$refs.anomalyMapData;
                var heatmapEl = document.querySelector('.compare-heatmap');
                if (anomalyEl && heatmapEl) {
                    var cleanup = setupHeatmapTooltip(anomalyEl, heatmapEl);
                    if (cleanup) registerCleanup(cleanup);
                }

                var imgEl = document.querySelector('.compare-heatmap');
                var containerEl = document.querySelector('.compare-container');
                if (containerEl && imgEl && self.resultData && self.resultData.bboxes) {
                    var bboxCleanups = setupBboxOverlays(containerEl, self.resultData.bboxes, imgEl);
                    bboxCleanups.forEach(function (c) { registerCleanup(c); });
                }
            },

            /** 取消推理 */
            cancelInference: function () {
                InferenceRunner.cancel();
                this.inferenceState = 'idle';
            },

            /** 重置：清除结果，返回上传状态 */
            resetInference: function () {
                InferenceRunner.cancel();
                runAllCleanups();
                if (this.uploadPreviewUrl) URL.revokeObjectURL(this.uploadPreviewUrl);
                this.uploadedFile = null;
                this.uploadPreviewUrl = null;
                this.resultData = null;
                this.inferenceState = 'idle';
                this.inferenceProgress = { stage: '', message: '', pct: 0 };
                this.errorMessage = '';

                // 重置文件 input
                var fileInput = this.$refs.fileInput;
                if (fileInput) fileInput.value = '';
            },

            // ─────────────────────────────────────────────
            // 算法卡片交叉观察（P2 流程图触发用）
            // ─────────────────────────────────────────────
            onCardIntersect: function () {
                // 将在 P2 中用于触发 SVG 流程图绘制动画
            }
        };
    });
});
