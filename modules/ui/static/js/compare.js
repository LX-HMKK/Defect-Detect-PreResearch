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
     * @param {File} imageFile - 上传的图片文件
     * @param {string} dataset - 数据集名称
     * @param {Object} callbacks
     * @param {Function} callbacks.onModelStart - 模型开始回调 ({model, name})
     * @param {Function} callbacks.onModelResult - 模型结果回调 (resultData)
     * @param {Function} callbacks.onModelError - 模型错误回调 ({model, name, message})
     * @param {Function} callbacks.onSummary - 排名摘要回调
     * @param {Function} callbacks.onDone - 完成回调
     * @param {Function} callbacks.onError - 全局错误回调 (message)
     */
    async run(imageFile, dataset, callbacks) {
        const { onModelStart, onModelResult, onModelError, onSummary, onDone, onError } = callbacks;

        // 取消已有请求
        if (this.abortController) {
            this.abortController.abort();
        }
        this.abortController = new AbortController();

        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('dataset', dataset);

        try {
            const response = await fetch('/api/compare', {
                method: 'POST',
                body: formData,
                signal: this.abortController.signal,
            });

            if (!response.ok) {
                onError('HTTP ' + response.status + ': ' + response.statusText);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

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
                            case 'model_start':
                                onModelStart(data);
                                break;
                            case 'model_result':
                                onModelResult(data);
                                break;
                            case 'model_error':
                                onModelError(data);
                                break;
                            case 'summary':
                                onSummary(data);
                                break;
                            case 'error':
                                onError(data.message || '对比请求失败');
                                return;
                            case 'done':
                                onDone();
                                return;
                        }
                    } catch (e) {
                        console.warn('[compare] SSE 数据解析失败:', dataStr);
                    }
                }
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                console.log('[compare] 请求已取消');
                return;
            }
            onError(err.message || '网络错误');
        }
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

            /** 模型顺序列表 */
            modelOrder: ['patchcore', 'padim', 'fre', 'draem'],

            /** 启动四模型对比 */
            startCompare() {
                // 检查是否有图片
                if (!window._compareImageFile) {
                    alert('请先在「单模型推理」区上传图片');
                    return;
                }

                // 重置所有槽位
                var self = this;
                Object.keys(this.compareSlots).forEach(function (k) {
                    self.compareSlots[k] = { status: 'pending', data: null, error: null };
                });
                this.compareRunning = true;
                this.compareDone = false;
                this.summary = null;

                var dataset = window._appDataset || 'bottle';

                CompareRunner.run(window._compareImageFile, dataset, {
                    onModelStart: function (data) {
                        self.compareSlots[data.model].status = 'active';
                    },
                    onModelResult: function (data) {
                        self.compareSlots[data.model] = { status: 'done', data: data, error: null };
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
        };
    });
});
