/**
 * TrainingRunner — SSE 训练客户端（基于 fetch ReadableStream，支持 POST 与中断）
 */
var TrainingRunner = {
    _abortController: null,

    run: function (payload, handlers) {
        var self = this;
        self.cancel();
        self._abortController = new AbortController();
        self._postStream('/api/train', payload, handlers, self._abortController.signal);
    },

    _postStream: function (url, payload, handlers, signal) {
        var self = this;
        fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: signal,
        }).then(function (response) {
            // 校验 HTTP 状态码，避免将 400/409 等错误响应当作 SSE 流解析
            if (!response.ok) {
                return response.text().then(function (text) {
                    if (handlers.onError) handlers.onError(text || ('HTTP ' + response.status));
                });
            }
            var reader = response.body.getReader();
            var decoder = new TextDecoder();
            var buffer = '';

            function read() {
                return reader.read().then(function (result) {
                    if (result.done) {
                        if (handlers.onDone) handlers.onDone();
                        return;
                    }
                    buffer += decoder.decode(result.value, { stream: true });
                    // SSE 规范使用 CRLF 换行，统一归一化为 LF 再解析
                    buffer = buffer.replace(/\r\n/g, '\n');
                    var lines = buffer.split('\n');
                    buffer = lines.pop();

                    var eventName = 'message';
                    var dataLines = [];
                    lines.forEach(function (line) {
                        if (line.startsWith('event:')) {
                            eventName = line.slice(6).trim();
                        } else if (line.startsWith('data:')) {
                            dataLines.push(line.slice(5).trim());
                        } else if (line === '') {
                            if (dataLines.length > 0) {
                                var data = JSON.parse(dataLines.join('\n'));
                                self._dispatch(eventName, data, handlers);
                            }
                            eventName = 'message';
                            dataLines = [];
                        }
                    });
                    return read();
                }).catch(function (err) {
                    if (err.name === 'AbortError') {
                        if (handlers.onDone) handlers.onDone();
                    } else if (handlers.onError) {
                        handlers.onError(String(err));
                    }
                });
            }
            return read();
        }).catch(function (err) {
            if (err.name !== 'AbortError' && handlers.onError) {
                handlers.onError(String(err));
            }
        });
    },

    _dispatch: function (event, data, handlers) {
        if (event === 'metric' && handlers.onMetric) handlers.onMetric(data);
        if (event === 'status' && handlers.onStatus) handlers.onStatus(data);
        if (event === 'completed' && handlers.onCompleted) handlers.onCompleted(data);
        if (event === 'error' && handlers.onError) handlers.onError(data.message || '训练失败');
    },

    cancel: function () {
        if (this._abortController) {
            this._abortController.abort();
            this._abortController = null;
        }
    },
};


document.addEventListener('alpine:init', function () {
    Alpine.data('training', function () {
        return {
            // 模型与算法
            models: [
                { key: 'patchcore', name: 'PatchCore', color: '#2997ff' },
                { key: 'padim', name: 'PaDiM', color: '#30d158' },
                { key: 'fre', name: 'FRE', color: '#ff9f0a' },
                { key: 'draem', name: 'DRAEM', color: '#bf5af2' },
            ],
            selectedModel: 'patchcore',

            // 数据集
            selectedDataset: '',
            trainSamples: [],
            trainSampleCount: 0,
            excludedSamples: [],

            // 参数
            epochs: 100,
            batchSize: 32,
            learningRate: 0.0001,
            seed: 42,
            showAdvanced: false,
            advancedParams: {
                patchcore: { coreset_sampling_ratio: 0.1, num_neighbors: 9 },
                padim: {},
                fre: { latent_dim: 220, pooling_kernel_size: 2 },
                draem: { beta_a: 0.1, beta_b: 1.0, enable_sspcab: false },
            },

            // 模型推荐 epoch
            modelDefaultEpochs: {
                patchcore: 1,
                padim: 1,
                fre: 50,
                draem: 100,
            },

            // 训练状态
            trainingState: 'idle', // idle | training | completed | error
            currentEpoch: 0,
            totalEpochs: 0,
            latestLoss: null,
            latestLR: null,
            latestAUROC: null,
            etaSeconds: null,
            errorMessage: '',
            metricsHistory: [],

            // 数字滚动的前一值缓存
            _prevEpoch: 0,
            _prevLoss: null,
            _prevLR: null,
            _prevAUROC: null,

            init: function () {
                var self = this;
                self.resetMonitor();
                self.epochs = self.modelDefaultEpochs[self.selectedModel] || 100;

                // 从全局 app 同步数据集，并监听后续变化（app 异步加载数据集）
                var app = self._getApp();
                if (app && app.selectedDataset) {
                    self.selectedDataset = app.selectedDataset;
                }
                if (app && app.$watch) {
                    app.$watch('selectedDataset', function (dataset) {
                        if (dataset !== self.selectedDataset) {
                            self.selectedDataset = dataset;
                        }
                    });
                }
                self.loadTrainSamples();

                self.$watch('selectedModel', function (model) {
                    self.epochs = self.modelDefaultEpochs[model] || 100;
                });
                self.$watch('selectedDataset', function (dataset) {
                    self.excludedSamples = [];
                    self.loadTrainSamples();
                    if (app) app.selectedDataset = dataset;
                });
                window.addEventListener('resize', function () {
                    self.$nextTick(function () { self.drawChart(); });
                });
            },

            _getApp: function () {
                return Alpine.store('app') || window.app;
            },

            loadTrainSamples: function () {
                var self = this;
                if (!self.selectedDataset) {
                    self.trainSamples = [];
                    self.trainSampleCount = 0;
                    return;
                }
                fetch('/api/train-samples?dataset=' + encodeURIComponent(self.selectedDataset))
                    .then(function (res) { return res.json(); })
                    .then(function (data) {
                        self.trainSamples = (data.samples || []).slice(0, 12);
                        self.trainSampleCount = data.total || 0;
                    })
                    .catch(function () {
                        self.trainSamples = [];
                        self.trainSampleCount = 0;
                    });
            },

            sampleUrl: function (sample) {
                return '/data/' + encodeURIComponent(this.selectedDataset) + '/' + sample;
            },

            sampleName: function (sample) {
                return sample.split('/').pop();
            },

            isExcluded: function (sample) {
                return this.excludedSamples.indexOf(this.sampleName(sample)) >= 0;
            },

            toggleExclude: function (sample) {
                if (this.trainingState === 'training') return;
                var name = this.sampleName(sample);
                var idx = this.excludedSamples.indexOf(name);
                if (idx >= 0) {
                    this.excludedSamples.splice(idx, 1);
                } else {
                    this.excludedSamples.push(name);
                }
            },

            get sampleCount() {
                return this.trainSampleCount;
            },

            get hasMetrics() {
                return this.metricsHistory.length > 0;
            },

            get statusText() {
                var map = {
                    idle: '就绪',
                    training: '训练中',
                    completed: '完成',
                    error: '错误',
                };
                return map[this.trainingState] || this.trainingState;
            },

            resetMonitor: function () {
                this.currentEpoch = 0;
                this.totalEpochs = 0;
                this.latestLoss = null;
                this.latestLR = null;
                this.latestAUROC = null;
                this.etaSeconds = null;
                this.metricsHistory = [];
                this._prevEpoch = 0;
                this._prevLoss = null;
                this._prevLR = null;
                this._prevAUROC = null;
                this._setMetricText('epochValue', '0');
                this._setMetricText('lossValue', '—');
                this._setMetricText('lrValue', '—');
                this._setMetricText('aurocValue', '—');
            },

            _setMetricText: function (refName, text) {
                var el = this.$refs[refName];
                if (el) el.textContent = text;
            },

            startTraining: function () {
                var self = this;
                if (!self.selectedDataset) return;

                self.resetMonitor();
                self._prevEpoch = 0;
                self._prevLoss = null;
                self._prevLR = null;
                self._prevAUROC = null;
                self.trainingState = 'training';

                TrainingRunner.run({
                    model: self.selectedModel,
                    dataset: self.selectedDataset,
                    epochs: parseInt(self.epochs, 10),
                    batch_size: parseInt(self.batchSize, 10),
                    learning_rate: parseFloat(self.learningRate),
                    seed: parseInt(self.seed, 10),
                    excluded_samples: self.excludedSamples,
                    advanced_params: self.advancedParams[self.selectedModel] || {},
                }, {
                    onMetric: function (data) {
                        self.totalEpochs = data.total_epochs;
                        if (data.epoch !== undefined && data.epoch !== null) {
                            self._rollNumber(self.$refs.epochValue, self._prevEpoch, data.epoch, function (v) { return String(Math.round(v)); });
                            self.currentEpoch = data.epoch;
                            self._prevEpoch = data.epoch;
                        }
                        if (data.train_loss !== undefined && data.train_loss !== null) {
                            var lossVal = parseFloat(data.train_loss);
                            self._rollNumber(self.$refs.lossValue, self._prevLoss === null ? lossVal : self._prevLoss, lossVal, function (v) { return v.toFixed(4); });
                            self.latestLoss = lossVal.toFixed(4);
                            self._prevLoss = lossVal;
                        }
                        if (data.learning_rate !== undefined && data.learning_rate !== null) {
                            var lrVal = parseFloat(data.learning_rate);
                            self._rollNumber(self.$refs.lrValue, self._prevLR === null ? lrVal : self._prevLR, lrVal, function (v) { return v.toExponential(2); });
                            self.latestLR = lrVal.toExponential(2);
                            self._prevLR = lrVal;
                        }
                        if (data.val_image_AUROC !== undefined && data.val_image_AUROC !== null) {
                            var aurocVal = parseFloat(data.val_image_AUROC);
                            self._rollNumber(self.$refs.aurocValue, self._prevAUROC === null ? aurocVal : self._prevAUROC, aurocVal, function (v) { return (v * 100).toFixed(1) + '%'; });
                            self.latestAUROC = (aurocVal * 100).toFixed(1) + '%';
                            self._prevAUROC = aurocVal;
                        }
                        if (data.eta_seconds !== undefined && data.eta_seconds !== null) self.etaSeconds = data.eta_seconds;
                        self.metricsHistory.push(data);
                        self.drawChart();
                    },
                    onStatus: function (data) {
                        // 可选：显示状态消息
                    },
                    onCompleted: function (data) {
                        self.trainingState = 'completed';
                        // 通知全局应用刷新模型/数据集列表
                        window.dispatchEvent(new CustomEvent('training-completed', {
                            detail: { model: self.selectedModel, category: self.selectedDataset }
                        }));
                    },
                    onError: function (msg) {
                        self.trainingState = 'error';
                        self.errorMessage = msg;
                    },
                });
            },

            /**
             * 数字滚动动画
             * @param {HTMLElement} el - 目标元素
             * @param {number} from - 起始值
             * @param {number} to - 目标值
             * @param {Function} formatter - 格式化函数
             */
            _rollNumber: function (el, from, to, formatter) {
                if (!el || from === to) return;
                var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
                if (reducedMotion) {
                    el.textContent = formatter(to);
                    return;
                }
                var duration = 500;
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

            stopTraining: function () {
                fetch('/api/train/stop', { method: 'POST' });
                TrainingRunner.cancel();
                this.trainingState = 'idle';
            },

            drawChart: function () {
                var canvas = this.$refs.trainingChart;
                if (!canvas) return;
                var ctx = canvas.getContext('2d');
                var dpr = window.devicePixelRatio || 1;
                var rect = canvas.getBoundingClientRect();
                canvas.width = rect.width * dpr;
                canvas.height = rect.height * dpr;
                ctx.scale(dpr, dpr);

                var w = rect.width;
                var h = rect.height;
                var padding = { top: 28, right: 44, bottom: 28, left: 44 };
                var self = this;

                // 从 CSS 变量读取当前主题颜色
                var style = getComputedStyle(canvas);
                var gridColor = style.getPropertyValue('--training-chart-grid').trim() || 'rgba(128,128,128,0.2)';
                var lossColor = style.getPropertyValue('--accent').trim() || '#5ac8fa';
                var aurocColor = style.getPropertyValue('--ok').trim() || '#30d158';
                var axisColor = style.getPropertyValue('--training-chart-axis').trim() || 'rgba(128,128,128,0.5)';
                var textColor = style.getPropertyValue('--text-secondary').trim() || 'rgba(128,128,128,0.7)';
                var isDark = document.documentElement.getAttribute('data-theme') === 'dark';

                ctx.clearRect(0, 0, w, h);

                var lossPoints = this.metricsHistory
                    .filter(function (m) { return m.train_loss !== undefined && m.train_loss !== null; })
                    .map(function (m) { return { epoch: m.epoch, value: m.train_loss }; });
                var aurocPoints = this.metricsHistory
                    .filter(function (m) { return m.val_image_AUROC !== undefined && m.val_image_AUROC !== null; })
                    .map(function (m) { return { epoch: m.epoch, value: m.val_image_AUROC }; });

                var chartW = w - padding.left - padding.right;
                var chartH = h - padding.top - padding.bottom;

                function drawGrid() {
                    ctx.strokeStyle = gridColor;
                    ctx.lineWidth = 1;
                    for (var i = 0; i <= 4; i++) {
                        var y = padding.top + chartH * i / 4;
                        ctx.beginPath();
                        ctx.moveTo(padding.left, y);
                        ctx.lineTo(w - padding.right, y);
                        ctx.stroke();
                    }
                }

                function drawSeries(points, color, minVal, maxVal, label, side) {
                    if (points.length === 0) return;
                    var range = Math.max(0.001, maxVal - minVal);

                    function coord(idx) {
                        var x = padding.left + chartW * (points.length > 1 ? idx / (points.length - 1) : 0.5);
                        var y = padding.top + chartH * (1 - (points[idx].value - minVal) / range);
                        return { x: x, y: y };
                    }

                    // 画线
                    if (points.length >= 2) {
                        ctx.strokeStyle = color;
                        ctx.lineWidth = isDark ? 2.5 : 2;
                        ctx.lineCap = 'round';
                        ctx.lineJoin = 'round';
                        ctx.shadowBlur = isDark ? 14 : 0;
                        ctx.shadowColor = isDark ? color + '88' : 'transparent';
                        ctx.beginPath();
                        points.forEach(function (p, idx) {
                            var c = coord(idx);
                            if (idx === 0) ctx.moveTo(c.x, c.y);
                            else ctx.lineTo(c.x, c.y);
                        });
                        ctx.stroke();
                        ctx.shadowBlur = 0;
                    }

                    // 画点（单点或最后一点高亮）
                    ctx.fillStyle = color;
                    points.forEach(function (p, idx) {
                        var c = coord(idx);
                        ctx.beginPath();
                        ctx.arc(c.x, c.y, points.length === 1 ? 5 : 3, 0, Math.PI * 2);
                        ctx.fill();
                    });

                    // 轴刻度
                    ctx.fillStyle = color;
                    ctx.font = '10px sans-serif';
                    ctx.textAlign = side === 'left' ? 'right' : 'left';
                    for (var i = 0; i <= 2; i++) {
                        var val = minVal + range * (1 - i / 2);
                        var y = padding.top + chartH * i / 2;
                        var text = val < 0.01 ? val.toExponential(1) : val.toFixed(3);
                        if (side === 'left') ctx.fillText(text, padding.left - 8, y + 3);
                        else ctx.fillText(text, w - padding.right + 8, y + 3);
                    }
                    ctx.textAlign = 'start';

                    // 图例
                    ctx.fillStyle = textColor;
                    ctx.font = '11px sans-serif';
                    var legendX = side === 'left' ? padding.left : w - padding.right - 80;
                    ctx.fillStyle = color;
                    ctx.fillRect(legendX, padding.top - 14, 10, 3);
                    ctx.fillStyle = textColor;
                    ctx.fillText(label, legendX + 14, padding.top - 10);
                }

                function drawNoCurveHint(message) {
                    ctx.fillStyle = textColor;
                    ctx.font = '13px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillText(message, w / 2, h / 2);
                    ctx.textAlign = 'start';
                }

                drawGrid();

                var hasLoss = lossPoints.length > 0;
                var hasAUROC = aurocPoints.length > 0;

                if (hasLoss) {
                    var lossMin = Math.min.apply(null, lossPoints.map(function (p) { return p.value; }));
                    var lossMax = Math.max.apply(null, lossPoints.map(function (p) { return p.value; }));
                    drawSeries(lossPoints, lossColor, lossMin, lossMax, 'Loss', 'left');
                }

                if (hasAUROC) {
                    var aurocMin = Math.min.apply(null, aurocPoints.map(function (p) { return p.value; }));
                    var aurocMax = Math.max.apply(null, aurocPoints.map(function (p) { return p.value; }));
                    // AUROC 通常落在 [0,1]，固定范围更易读
                    drawSeries(aurocPoints, aurocColor, Math.min(0.0, aurocMin), 1.0, 'val AUROC', 'right');
                }

                // 没有任何可绘制序列时，给出明确提示
                if (!hasLoss && !hasAUROC) {
                    var noCurveModels = { patchcore: true, padim: true };
                    if (noCurveModels[self.selectedModel]) {
                        drawNoCurveHint('PatchCore / PaDiM 无需训练，无训练曲线');
                    } else if (self.metricsHistory.length > 0) {
                        drawNoCurveHint('当前指标暂无可用曲线数据');
                    }
                }

                // 底部 epoch 轴
                ctx.fillStyle = axisColor;
                ctx.font = '10px sans-serif';
                ctx.textAlign = 'center';
                var total = this.totalEpochs || 1;
                for (var i = 0; i <= 4; i++) {
                    var epoch = Math.round(total * i / 4);
                    var x = padding.left + chartW * i / 4;
                    ctx.fillText(String(epoch), x, h - padding.bottom + 14);
                }
                ctx.textAlign = 'start';
            },
        };
    });
});
