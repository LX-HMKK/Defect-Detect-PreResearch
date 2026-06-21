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

                // 暴露全局引用，供子作用域（如 compare）访问
                Alpine.store('app', self);
                window.app = self;

                // 获取模型列表
                self.fetchModels();

                // 切换模型时刷新自训练模型列表
                self.$watch('selectedModel', function () {
                    self.loadSelfTrainedModels();
                });

                // 切换推理来源时刷新测试图片
                self.$watch('inferenceSource', function () {
                    self.loadTestImages();
                });

                // 设置滚动监听（用于导航点更新）
                self.setupScrollObserver();

                // 启动健康检查轮询
                self.startHealthCheck();

                // 训练完成后刷新模型/数据集列表
                window.addEventListener('training-completed', function () {
                    self.fetchModels();
                });

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
            snapProgress: 0,  // 0.0 ~ 1.0，表示两页之间的滚动进度（驱动进度环填充）
            sectionCount: 4,

            /** 导航点 tooltip 标签 */
            sectionNames: ['算法介绍', '训练工作室', '单模型推理', '四模型对比'],

            setupScrollObserver: function () {
                var self = this;
                var container = self.$refs.snapContainer;
                if (!container) {
                    // 降级：使用 window scroll（兼容旧 HTML 或非 snap 模式）
                    container = window;
                }

                // ── 导航栏加深（基于滚动位置）──
                var scrollHandler = function () {
                    var scrollY = container === window ? window.scrollY : container.scrollTop;
                    self.scrolled = scrollY > 50;
                };

                if (container === window) {
                    window.addEventListener('scroll', scrollHandler, { passive: true });
                } else {
                    container.addEventListener('scroll', scrollHandler, { passive: true });
                }

                // ── IntersectionObserver 检测当前 section ──
                // 通过 class 选择所有 .snap-page，避免嵌套 x-data 导致 $refs 不可见
                var sections = container === window
                    ? Array.prototype.slice.call(document.querySelectorAll('.snap-page'))
                    : Array.prototype.slice.call(container.querySelectorAll('.snap-page'));

                if (sections.length === 0) return;

                // 同步页码总数（label 使用）
                self.sectionCount = sections.length;

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

                        if (maxRatio > 0 && maxIdx !== self.currentSection) {
                            var prevSection = sections[self.currentSection];
                            var nextSection = sections[maxIdx];

                            // 1. 先触发旧 section 的退出动画（向下推出，与滚动方向一致）
                            if (prevSection && window.Anim && window.Anim.snapPageExit) {
                                window.Anim.snapPageExit(prevSection);
                            }

                            // 2. 更新 currentSection
                            self.currentSection = maxIdx;

                            // 3. 触发新 section 的入场动画（snapPageEnter 内部会移除 exiting class + 取消旧动画）
                            if (nextSection && window.Anim && window.Anim.snapPageEnter) {
                                window.Anim.snapPageEnter(nextSection, { staggerMs: 80, duration: 500 });
                            }
                        }

                        // 计算 snap 进度（用于进度环填充）
                        if (container !== window) {
                            var totalHeight = container.scrollHeight - container.clientHeight;
                            if (totalHeight > 0) {
                                self.snapProgress = Math.min(1, container.scrollTop / totalHeight);
                            }
                        }
                    },
                    {
                        threshold: [0, 0.15, 0.3, 0.5, 0.7, 0.85, 1],
                        root: container === window ? null : container  // 关键修复：以 snap-container 为观察根
                    }
                );

                sections.forEach(function (s) {
                    observer.observe(s);
                });

                // ── 滚动事件更新进度环（高频率更新，保证进度环流畅）──
                if (container !== window) {
                    container.addEventListener('scroll', function () {
                        var totalHeight = container.scrollHeight - container.clientHeight;
                        if (totalHeight > 0) {
                            self.snapProgress = Math.min(1, container.scrollTop / totalHeight);
                        }
                    }, { passive: true });
                }

                // ── 键盘导航（↑↓ 切换 section）──
                window.addEventListener('keydown', function (e) {
                    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
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
                var container = this.$refs.snapContainer;
                var sections = container
                    ? Array.prototype.slice.call(container.querySelectorAll('.snap-page'))
                    : Array.prototype.slice.call(document.querySelectorAll('.snap-page'));

                var target = sections[idx];
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
            datasets: [],          // 对象列表，如 [{value:"default/bottle", label:"bottle", source:"default"}, ...]
            selectedDataset: '',   // value 字符串，如 "default/bottle"
            models: [],
            selectedModel: 'patchcore',

            get defaultDatasets() {
                return this.datasets.filter(function (d) { return d.source === 'default'; });
            },

            get userDatasets() {
                return this.datasets.filter(function (d) { return d.source === 'user'; });
            },

            get datasetDisplayName() {
                var ds = this.datasets.find(function (d) { return d.value === this.selectedDataset; }, this);
                return ds ? ds.label : (this.selectedDataset || '数据集');
            },

            fetchModels: async function () {
                try {
                    var res = await fetch('/api/models');
                    var data = await res.json();
                    this.models = data.models || [];
                    this.datasets = data.datasets || [];
                    // 优先选中默认第一个
                    if (this.datasets.length > 0) {
                        var defaultFirst = this.datasets.find(function (d) { return d.source === 'default'; });
                        this.selectedDataset = defaultFirst ? defaultFirst.value : this.datasets[0].value;
                    }
                    this.loadSelfTrainedModels();
                    this.loadTestImages();
                } catch (e) {
                    console.warn('[app] 获取模型列表失败:', e);
                }
            },

            loadTestImages: function () {
                var self = this;
                var dataset = self.inferenceSource === 'pretrained'
                    ? self.selectedDataset
                    : (self.selectedSelfTrainedModel ? self.selectedSelfTrainedModel.category : '');

                if (!dataset) {
                    self.testImages = [];
                    self.selectedTestImage = '';
                    self.testImagePreviewUrl = '';
                    return;
                }

                fetch('/api/test-images?dataset=' + encodeURIComponent(dataset))
                    .then(function (res) { return res.json(); })
                    .then(function (data) {
                        self.testImages = data.images || [];
                        self.selectedTestImage = self.testImages[0] || '';
                        self.updateTestImagePreview();
                    })
                    .catch(function (e) {
                        console.warn('[app] 加载测试图片失败:', e);
                        self.testImages = [];
                        self.selectedTestImage = '';
                        self.testImagePreviewUrl = '';
                    });
            },

            updateTestImagePreview: function () {
                var self = this;
                if (!self.selectedTestImage) {
                    self.testImagePreviewUrl = '';
                    return;
                }
                var dataset = self.inferenceSource === 'pretrained'
                    ? self.selectedDataset
                    : (self.selectedSelfTrainedModel ? self.selectedSelfTrainedModel.category : '');
                if (!dataset) {
                    self.testImagePreviewUrl = '';
                    return;
                }
                self.testImagePreviewUrl = '/data/' + dataset + '/' + self.selectedTestImage;
            },

            loadSelfTrainedModels: function () {
                var self = this;
                fetch('/api/self-trained-models?model=' + encodeURIComponent(self.selectedModel))
                    .then(function (res) { return res.json(); })
                    .then(function (data) {
                        self.selfTrainedModels = data.models || [];
                        if (!self.selfTrainedModels.find(function (m) {
                            return self.selectedSelfTrainedModel && m.path === self.selectedSelfTrainedModel.path;
                        })) {
                            self.selectedSelfTrainedModel = self.selfTrainedModels[0] || null;
                        }
                        if (self.inferenceSource === 'self_trained') {
                            self.syncDatasetFromSelfTrained();
                            self.loadTestImages();
                        }
                    })
                    .catch(function (e) {
                        console.warn('[app] 加载自训练模型失败:', e);
                        self.selfTrainedModels = [];
                        self.selectedSelfTrainedModel = null;
                    });
            },

            syncDatasetFromSelfTrained: function () {
                if (this.selectedSelfTrainedModel) {
                    this.selectedDataset = this.selectedSelfTrainedModel.category;
                }
            },

            // ─────────────────────────────────────────────
            // 推理状态
            // ─────────────────────────────────────────────
            inferenceState: 'idle',  // idle | loading | inferring | done | error
            inferenceProgress: { stage: '', message: '', pct: 0 },
            resultData: {},
            errorMessage: '',

            // 推理来源
            inferenceSource: 'pretrained', // 'pretrained' | 'self_trained'
            selfTrainedModels: [],
            selectedSelfTrainedModel: null,

            // 图片选择（从 test/）
            testImages: [],
            selectedTestImage: '',
            testImagePreviewUrl: '',

            /** 开始推理 */
            startInference: function () {
                var self = this;
                if (!self.selectedTestImage) return;
                if (self.inferenceState === 'loading' || self.inferenceState === 'inferring') return;

                // 清理上一次的可视化元素
                runAllCleanups();

                self.inferenceState = 'loading';
                self.inferenceProgress = { stage: 'loading_model', message: '正在加载模型...', pct: 10 };

                var payload = {
                    model: self.selectedModel,
                    dataset: self.inferenceSource === 'pretrained'
                        ? self.selectedDataset
                        : self.selectedSelfTrainedModel.category,
                    image: self.selectedTestImage,
                    source: self.inferenceSource,
                };
                if (self.inferenceSource === 'self_trained') {
                    payload.self_trained_path = self.selectedSelfTrainedModel.path;
                }

                InferenceRunner.run('/api/predict', payload, {
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
                            // Apple 风格结果仪表盘揭示动画
                            setTimeout(function () {
                                var dashboard = document.querySelector('.result-dashboard');
                                if (dashboard && window.Anim && window.Anim.resultReveal) {
                                    window.Anim.resultReveal(dashboard);
                                }
                            }, 80);
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

            /** 重置：清除结果 */
            resetInference: function () {
                InferenceRunner.cancel();
                runAllCleanups();
                this.resultData = null;
                this.inferenceState = 'idle';
                this.inferenceProgress = { stage: '', message: '', pct: 0 };
                this.errorMessage = '';
            },

        };
    });
});
