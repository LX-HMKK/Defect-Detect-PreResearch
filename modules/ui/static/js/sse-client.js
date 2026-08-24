/**
 * Shared SSE streaming client — /api/predict 与 /api/compare 共用同一套
 * fetch + EventSource 协议解析（CRLF 归一、\n\n 切块、event:/data: 解析）。
 *
 * 三个后端端点（predict / compare / train）都以 sse-starlette 协议输出单行 JSON，
 * 因此可参数化为一个事件→handler 映射驱动。training.js 因采用不同的 promise 链式
 * 结构与行级 split，暂不复用本模块。
 */

window.SSEClient = {
    /**
     * 发起 SSE POST 请求并消费事件流。
     * @param {string} url - API 端点（如 /api/predict）
     * @param {Object} payload - JSON 请求体
     * @param {AbortSignal} signal - 取消信号（来自 AbortController）
     * @param {Object} handlers - { eventType: (data) => boolean }；handler 返回 false 表示终止流
     * @param {Object} opts - { tag, onHttpError, onTransportError }
     */
    async run(url, payload, signal, handlers, opts) {
        const tag = (opts && opts.tag) || 'sse';
        const onHttpError = ((opts && opts.onHttpError) || function () {});
        const onTransportError = ((opts && opts.onTransportError) || function () {});

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: signal,
            });

            if (!response.ok) {
                const text = await response.text().catch(function () { return ''; });
                onHttpError('HTTP ' + response.status + ': ' + (text || response.statusText));
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // 归一化行尾：\r\n → \n（sse-starlette 使用 CRLF，但 JS 解析器期望 LF）
                buffer = buffer.replace(/\r\n/g, '\n');

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
                        const fn = (handlers && handlers[eventType]);
                        if (fn) {
                            const keepGoing = fn(data);
                            if (keepGoing === false) {
                                return;
                            }
                        }
                    } catch (e) {
                        console.warn('[' + tag + '] SSE 数据解析失败:', dataStr);
                    }
                }
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                console.log('[' + tag + '] 请求已取消');
                return;
            }
            onTransportError(err.message || '网络错误');
        }
    }
};
