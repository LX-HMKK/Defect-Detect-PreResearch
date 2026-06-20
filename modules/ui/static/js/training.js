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

            // 样本
            samples: [],
            isDragOver: false,
            sessionId: null,
            datasetPath: null,
            category: null,

            // 参数
            epochs: 100,
            batchSize: 32,
            learningRate: 0.0001,
            seed: 42,

            // 训练状态
            trainingState: 'idle', // idle | uploading | training | completed | error
            currentEpoch: 0,
            totalEpochs: 0,
            latestLoss: null,
            latestLR: null,
            latestAUROC: null,
            etaSeconds: null,
            errorMessage: '',
            metricsHistory: [],

            init: function () {
                this.resetMonitor();
                var self = this;
                window.addEventListener('resize', function () {
                    self.$nextTick(function () { self.drawChart(); });
                });
            },

            get sampleCount() {
                return this.samples.filter(function (s) { return !s.excluded; }).length;
            },

            get hasMetrics() {
                return this.metricsHistory.length > 0;
            },

            get statusText() {
                var map = {
                    idle: '就绪',
                    uploading: '上传中',
                    training: '训练中',
                    completed: '完成',
                    error: '错误',
                };
                return map[this.trainingState] || this.trainingState;
            },

            onSelectSamples: function (event) {
                this._uploadFiles(event.target.files);
            },

            onDropSamples: function (event) {
                this.isDragOver = false;
                this._uploadFiles(event.dataTransfer.files);
            },

            _uploadFiles: function (fileList) {
                var self = this;
                var files = Array.from(fileList).filter(function (f) { return f.type.startsWith('image/'); });
                if (files.length === 0) return;

                self.trainingState = 'uploading';
                var form = new FormData();
                files.forEach(function (f) { form.append('files', f); });

                fetch('/api/upload-samples', {
                    method: 'POST',
                    body: form,
                }).then(function (res) {
                    return res.json();
                }).then(function (data) {
                    self.sessionId = data.session_id;
                    self.datasetPath = data.dataset_path;
                    self.category = data.category;
                    self.samples = data.samples.map(function (name) {
                        return {
                            name: name,
                            url: '/uploads/' + self.category + '/train/good/' + name,
                            excluded: false,
                        };
                    });
                    self.trainingState = 'idle';
                }).catch(function (err) {
                    self.trainingState = 'error';
                    self.errorMessage = String(err);
                });
            },

            toggleExclude: function (idx) {
                this.samples[idx].excluded = !this.samples[idx].excluded;
            },

            resetMonitor: function () {
                this.currentEpoch = 0;
                this.totalEpochs = 0;
                this.latestLoss = null;
                this.latestLR = null;
                this.latestAUROC = null;
                this.etaSeconds = null;
                this.metricsHistory = [];
            },

            startTraining: function () {
                var self = this;
                if (!self.datasetPath) return;

                self.resetMonitor();
                self.trainingState = 'training';

                TrainingRunner.run({
                    model: self.selectedModel,
                    dataset_path: self.datasetPath,
                    category: self.category,
                    epochs: parseInt(self.epochs, 10),
                    batch_size: parseInt(self.batchSize, 10),
                    learning_rate: parseFloat(self.learningRate),
                    seed: parseInt(self.seed, 10),
                }, {
                    onMetric: function (data) {
                        self.currentEpoch = data.epoch;
                        self.totalEpochs = data.total_epochs;
                        if (data.train_loss !== undefined) self.latestLoss = data.train_loss.toFixed(4);
                        if (data.learning_rate !== undefined) self.latestLR = data.learning_rate.toExponential(2);
                        if (data.val_image_AUROC !== undefined) self.latestAUROC = (data.val_image_AUROC * 100).toFixed(1) + '%';
                        if (data.eta_seconds !== undefined) self.etaSeconds = data.eta_seconds;
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
                            detail: { model: self.selectedModel, category: self.category }
                        }));
                    },
                    onError: function (msg) {
                        self.trainingState = 'error';
                        self.errorMessage = msg;
                    },
                });
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
                var padding = 30;

                ctx.clearRect(0, 0, w, h);

                if (this.metricsHistory.length < 2) return;

                var losses = this.metricsHistory
                    .filter(function (m) { return m.train_loss !== undefined && m.train_loss !== null; })
                    .map(function (m) { return m.train_loss; });
                if (losses.length < 2) return;

                var maxLoss = Math.max.apply(null, losses);
                var minLoss = Math.min.apply(null, losses);
                var range = Math.max(0.001, maxLoss - minLoss);

                // 网格线
                ctx.strokeStyle = 'rgba(255,255,255,0.06)';
                ctx.lineWidth = 1;
                for (var i = 0; i <= 4; i++) {
                    var y = padding + (h - 2 * padding) * i / 4;
                    ctx.beginPath();
                    ctx.moveTo(padding, y);
                    ctx.lineTo(w - padding, y);
                    ctx.stroke();
                }

                // Loss 曲线
                ctx.strokeStyle = '#5ac8fa';
                ctx.lineWidth = 2;
                ctx.beginPath();
                losses.forEach(function (loss, idx) {
                    var x = padding + (w - 2 * padding) * idx / (losses.length - 1);
                    var y = padding + (h - 2 * padding) * (1 - (loss - minLoss) / range);
                    if (idx === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.stroke();

                // 轴标签
                ctx.fillStyle = 'rgba(255,255,255,0.4)';
                ctx.font = '10px sans-serif';
                ctx.fillText('Loss', padding, padding - 8);
                ctx.fillText('Epoch', w - padding - 28, h - padding + 16);
            },
        };
    });
});
