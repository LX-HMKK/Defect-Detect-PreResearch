---
name: unresolved-inference-bug
description: 推理流程仍有未解决问题 — 点击推理后进度卡在"正在加载模型…"，最终静默失败
metadata:
  type: project
---

Phase 2 推理流程已修复两个根因（Alpine x-if null 渲染 + SSE asyncio.to_thread 阻塞），但 Playwright 实测仍然失败：

- 上传图片后点击"开始推理"，按钮变"推理中…"（disabled），进度显示"正在加载模型…"
- 45 秒后无结果面板出现，error-card 出现（仅显示 ⚠ 图标，无具体错误消息）
- 浏览器 console：0 errors, 4 warnings（页面本身无 JS 错误）

**可能原因：**
1. 模型 checkpoint 不存在或路径解析失败 → `detector.load_model()` 抛 ValueError → SSE 发 error 事件
2. `_run_prediction()` 内部异常被 catch 但错误消息未正确传递到前端
3. 前端 `onError` 回调设置了 `errorMessage` 但 error-card 的 `x-text` 绑定了不存在的属性
4. `resolve_project_path` 在 worktree 删除后的路径问题

**如何复现：**
```bash
cd D:/StudyWorks/3.2/Defect-Detect-PreResearch
python scripts/run_ui.py --port 8765 --no-browser
# 浏览器打开 http://127.0.0.1:8765
# 上传 test image: data/bottle/test/broken_large/000.png
# 选择 PatchCore，点击"开始推理"
# 观察：进度卡住，最终显示空错误卡片
```

**调试入口：**
1. `modules/ui/server.py:337` — `/api/predict` 的 `_run_prediction()` 调用
2. `modules/ui/server.py:178` — `detector.load_model()` 可能失败
3. `modules/ui/static/js/inference.js:92-94` — SSE error 事件处理
4. `modules/ui/static/js/app.js:342-345` — onError 回调
5. `modules/ui/static/index.html:468-471` — error-card 模板
6. 浏览器 DevTools Network 标签 → 查看 `/api/predict` 的 SSE 响应内容
