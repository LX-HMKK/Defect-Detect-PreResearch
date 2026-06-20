/**
 * Hero 视觉动画 — 检测管线 SVG 路径描边 + 节点呼吸
 *
 * 暴露 window.HeroVisual = { play, stop }
 * 在 alpine:initialized 后自动播放一次
 */
(function() {
    'use strict';

    var _svg = null;
    var _paths = [];
    var _nodes = [];
    var _isPlaying = false;
    var _prefersReduced = false;

    function _detectReducedMotion() {
        return window.matchMedia &&
            window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    function _getElements() {
        _svg = document.querySelector('.hero-visual');
        if (!_svg) return false;
        _paths = Array.prototype.slice.call(_svg.querySelectorAll('.hv-path'));
        _nodes = Array.prototype.slice.call(_svg.querySelectorAll('.hv-node'));
        return _paths.length > 0;
    }

    function _resetPaths() {
        _paths.forEach(function(path) {
            path.style.strokeDashoffset = '900';
            path.style.opacity = '1';
        });
        _nodes.forEach(function(node) {
            var rect = node.querySelector('rect');
            if (rect) {
                rect.style.stroke = '';
                rect.style.fill = '';
            }
        });
    }

    function play() {
        if (_isPlaying) return;
        _prefersReduced = _detectReducedMotion();
        if (!_getElements()) return;

        if (_prefersReduced) {
            _paths.forEach(function(path) {
                path.style.strokeDashoffset = '0';
            });
            return;
        }

        _isPlaying = true;
        _svg.classList.add('is-playing');
        _resetPaths();

        // 强制重排确保动画从头开始
        void _svg.offsetWidth;

        // 路径描边动画由 CSS @keyframes hvDraw 驱动
        // 节点呼吸由 CSS @keyframes hvNodePulse 驱动
        // 此处仅负责添加 is-playing 类并管理状态

        // 2.2s 后描边完成，进入脉冲阶段
        setTimeout(function() {
            // 脉冲阶段已自动由 CSS animation 处理
        }, 2200);
    }

    function stop() {
        _isPlaying = false;
        if (_svg) {
            _svg.classList.remove('is-playing');
        }
        if (_paths.length) {
            _paths.forEach(function(path) {
                path.style.animation = 'none';
                path.style.strokeDashoffset = '900';
            });
        }
        if (_nodes.length) {
            _nodes.forEach(function(node) {
                var rect = node.querySelector('rect');
                if (rect) {
                    rect.style.animation = 'none';
                }
            });
        }
    }

    // 暴露全局 API
    window.HeroVisual = {
        play: play,
        stop: stop
    };

    // alpine:initialized 后自动播放一次
    document.addEventListener('alpine:initialized', function() {
        // 延迟一小段时间确保 DOM 已渲染
        setTimeout(play, 300);
    });

    // 若 Alpine 已初始化（脚本后加载），立即尝试
    if (window.Alpine && window.Alpine.version) {
        setTimeout(play, 300);
    }
})();
