/**
 * Four-model comparison runner — SSE 流式客户端 + Alpine 组件
 *
 * 包含:
 *   1. CompareRunner — SSE 事件流消费器（/api/compare）
 *   2. compare — Alpine 组件：四列并排对比 UI
 */

/* ═══════════════════════════════════════════════════════════════════════════
   1. SSE 对比客户端
   ═══════════════════════════════════════════════════════════════════════════ */
const CompareRunner = {
    abortController: null,

    /**
     * 发起 SSE 四模型对比请求
     * @param {string} url - API 端点路径（如 /api/compare）
     * @param {Object} payload - JSON 请求体，包含 dataset 与 image
     * @param {Object} callbacks
     * @param {Function} callbacks.onModelStart - 模型开始回调 ({model, name})
     * @param {Function} callbacks.onModelResult - 模型结果回调 (resultData)
     * @param {Function} callbacks.onModelError - 模型错误回调 ({model, name, message})
     * @param {Function} callbacks.onSummary - 排名摘要回调
     * @param {Function} callbacks.onDone - 完成回调
     * @param {Function} callbacks.onError - 全局错误回调 (message)
     */
    async run(url, payload, callbacks) {
        const { onModelStart, onModelResult, onModelError, onSummary, onDone, onError } = callbacks;

        // 取消已有请求
        if (this.abortController) {
            this.abortController.abort();
        }
        this.abortController = new AbortController();

        // 共享 SSE 客户端（sse-client.js）处理 fetch + CRLF 归一 + \n\n 切块 +
        // event:/data: 解析，按事件分发到 handler。handler 返回 false 表示终止流。
        await SSEClient.run(url, payload, this.abortController.signal, {
            model_start:  function (data) { onModelStart(data); return true; },
            model_result: function (data) { onModelResult(data); return true; },
            model_error:  function (data) { onModelError(data); return true; },
            summary:      function (data) { onSummary(data); return true; },
            error:        function (data) { onError(data.message || '对比请求失败'); return false; },
            done:         function () { onDone(); return false; },
        }, {
            tag: 'compare',
            onHttpError: onError,
            onTransportError: onError,
        });
    },

    /** 取消当前对比请求 */
    cancel() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
    }
};


/* ═══════════════════════════════════════════════════════════════════════════
   2. Alpine 组件 — 四模型对比 UI
   ═══════════════════════════════════════════════════════════════════════════ */
document.addEventListener('alpine:init', function () {
    Alpine.data('compare', function () {
        return {
            // 每个模型槽位: { status: 'pending'|'active'|'done'|'error', data: null, error: null }
            compareSlots: {
                patchcore: { status: 'pending', data: null, error: null },
                padim:    { status: 'pending', data: null, error: null },
                fre:      { status: 'pending', data: null, error: null },
                draem:    { status: 'pending', data: null, error: null },
            },
            compareRunning: false,
            compareDone: false,
            summary: null,  // { best_model, best_name, best_score, ranking: [...] }

            /** 模型显示名映射 */
            modelNames: {
                patchcore: 'PatchCore',
                padim: 'PaDiM',
                fre: 'FRE',
                draem: 'DRAEM',
            },

            /** 模型简介 */
            modelDesc: {
                patchcore: 'CNN 特征记忆库 + 最近邻搜索',
                padim: 'CNN 多尺度特征 + 马氏距离',
                fre: 'ResNet50 特征重构误差',
                draem: '合成异常增强 + 判别网络',
            },

            /** 模型图标 */
            modelIcons: {
                patchcore: '&#128269;',
                padim: '&#128202;',
                fre: '&#128260;',
                draem: '&#127919;',
            },

            /** 模型色标 */
            modelColors: {
                patchcore: '#2997ff',
                padim: '#30d158',
                fre: '#ff9f0a',
                draem: '#bf5af2',
            },

            /** 模型顺序列表 */
            modelOrder: ['patchcore', 'padim', 'fre', 'draem'],

            /** 获取全局 app 实例 */
            _getApp: function () {
                return Alpine.store('app') || window.app;
            },

            /** 启动四模型对比 */
            startCompare() {
                var app = this._getApp();
                var dataset = app ? app.selectedDataset : '';
                var image = app ? app.selectedTestImage : '';

                if (!dataset || !image) {
                    alert('请先在「单模型推理」区选择数据集与测试图片');
                    return;
                }

                // 重置所有槽位
                var self = this;
                Object.keys(this.compareSlots).forEach(function (k) {
                    self.compareSlots[k] = { status: 'pending', data: null, error: null };
                    self._resetSlotNumbers(k);
                });
                this.compareRunning = true;
                this.compareDone = false;
                this.summary = null;

                CompareRunner.run('/api/compare', { dataset: dataset, image: image }, {
                    onModelStart: function (data) {
                        self.compareSlots[data.model].status = 'active';
                    },
                    onModelResult: function (data) {
                        self.compareSlots[data.model] = { status: 'done', data: data, error: null };
                        self.$nextTick(function () {
                            self._rollCompareNumber('compare-score-' + data.model, 0, data.score, function (v) { return v.toFixed(4); });
                            self._rollCompareNumber('compare-confidence-' + data.model, 0, data.confidence * 100, function (v) { return v.toFixed(1) + '%'; });
                            setTimeout(function () {
                                self.setupCompareBbox(data.model);
                            }, 200);
                        });
                    },
                    onModelError: function (data) {
                        self.compareSlots[data.model] = { status: 'error', data: null, error: data.message };
                    },
                    onSummary: function (data) {
                        self.summary = data;
                    },
                    onDone: function () {
                        self.compareRunning = false;
                        self.compareDone = true;
                        self.$nextTick(function () {
                            setTimeout(function () {
                                var wall = document.querySelector('.compare-wall');
                                if (wall && window.Anim && window.Anim.compareReveal) {
                                    window.Anim.compareReveal(wall);
                                }
                            }, 120);
                        });
                    },
                    onError: function (msg) {
                        self.compareRunning = false;
                        alert('对比失败: ' + msg);
                    }
                });
            },

            /** 获取槽位 CSS class */
            getSlotClass(modelKey) {
                var slot = this.compareSlots[modelKey];
                if (slot.status === 'active') return 'compare-slot--active';
                if (slot.status === 'done') return 'compare-slot--done';
                if (slot.status === 'error') return 'compare-slot--error';
                return '';
            },

            /** 获取槽位模型显示名 */
            getModelName(modelKey) {
                return this.modelNames[modelKey] || modelKey;
            },

            /** 获取槽位模型简介 */
            getModelDesc(modelKey) {
                return this.modelDesc[modelKey] || '';
            },

            /** 获取槽位模型图标 */
            getModelIcon(modelKey) {
                return this.modelIcons[modelKey] || '';
            },

            /** 获取槽位模型色标 */
            getModelColor(modelKey) {
                return this.modelColors[modelKey] || 'var(--accent)';
            },

            /** 已完成模型计数 */
            get completedCount() {
                var self = this;
                return Object.values(this.compareSlots).filter(function (s) {
                    return s.status === 'done' || s.status === 'error';
                }).length;
            },

            // 状态查询方法
            slotIsDone:    function (mk) { return this.compareSlots[mk].status === 'done'; },
            slotIsActive:  function (mk) { return this.compareSlots[mk].status === 'active'; },
            slotIsPending: function (mk) { return this.compareSlots[mk].status === 'pending'; },
            slotIsError:   function (mk) { return this.compareSlots[mk].status === 'error'; },

            /** 为对比槽位设置 bbox overlays（热力图加载完成后调用） */
            setupCompareBbox: function (mk) {
                var wrap = document.getElementById('compare-wrap-' + mk);
                var img  = document.getElementById('compare-heatmap-' + mk);
                if (!wrap || !img) return;

                var bboxes;
                try {
                    bboxes = JSON.parse(wrap.getAttribute('data-bboxes') || '[]');
                } catch (e) { return; }
                if (!bboxes || bboxes.length === 0) return;

                // 清除已有 overlay
                wrap.querySelectorAll('.bbox-overlay').forEach(function (el) { el.remove(); });

                bboxes.forEach(function (bbox, i) {
                    var div = document.createElement('div');
                    div.className = 'bbox-overlay bbox-overlay-' + i;
                    div.title = '缺陷区域 · 得分: ' + bbox[4].toFixed(4);
                    wrap.appendChild(div);
                });

                // 定位更新函数（复用 object-fit: contain 逻辑）
                var update = function () {
                    var imgRect = img.getBoundingClientRect();
                    var wrapRect = wrap.getBoundingClientRect();

                    var naturalW = img.naturalWidth;
                    var naturalH = img.naturalHeight;
                    var displayW = imgRect.width;
                    var displayH = imgRect.height;

                    var scale = Math.min(displayW / naturalW, displayH / naturalH);
                    var renderedW = naturalW * scale;
                    var renderedH = naturalH * scale;
                    var contentX = (displayW - renderedW) / 2;
                    var contentY = (displayH - renderedH) / 2;
                    var offsetX = imgRect.left - wrapRect.left;
                    var offsetY = imgRect.top - wrapRect.top;

                    bboxes.forEach(function (bb, i) {
                        var ov = wrap.querySelector('.bbox-overlay-' + i);
                        if (!ov) return;
                        ov.style.left   = (offsetX + contentX + bb[0] * scale) + 'px';
                        ov.style.top    = (offsetY + contentY + bb[1] * scale) + 'px';
                        ov.style.width  = (bb[2] * scale) + 'px';
                        ov.style.height = (bb[3] * scale) + 'px';
                    });
                };

                update();
                // 监听 resize / 图片加载变化
                var ro = new ResizeObserver(update);
                ro.observe(img);
                img.addEventListener('load', update);
            },

            _resetSlotNumbers: function (mk) {
                var scoreEl = document.getElementById('compare-score-' + mk);
                var confEl = document.getElementById('compare-confidence-' + mk);
                if (scoreEl) scoreEl.textContent = '0.0000';
                if (confEl) confEl.textContent = '0.0%';
            },

            /**
             * 对比得分数字滚动动画
             * @param {string} id - 目标元素 ID
             * @param {number} from - 起始值
             * @param {number} to - 目标值
             * @param {Function} formatter - 格式化函数
             */
            _rollCompareNumber: function (id, from, to, formatter) {
                var el = document.getElementById(id);
                if (!el || from === to) return;
                var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
                if (reducedMotion) {
                    el.textContent = formatter(to);
                    return;
                }
                var duration = 700;
                var startTime = performance.now();
                el.classList.add('is-rolling');
                function step(now) {
                    var t = Math.min(1, (now - startTime) / duration);
                    var eased = 1 - Math.pow(1 - t, 3);
                    var val = from + (to - from) * eased;
                    el.textContent = formatter(val);
                    if (t < 1) {
                        requestAnimationFrame(step);
                    } else {
                        el.classList.remove('is-rolling');
                    }
                }
                requestAnimationFrame(step);
            },
        };
    });
});
