/**
 * 算法卡片横向轮播 — AirPods Pro 风格
 *
 * 功能：
 * - 横向 scroll-snap 滚动，每次吸附一张卡片
 * - 当前卡片高亮，非当前卡片弱化（scale/opacity/blur）
 * - 指示点同步、01/04 计数器
 * - 键盘 ←→ 导航、按钮前后切换
 * - 无障碍：prefers-reduced-motion 时禁用滚动动画
 */
(function() {
    'use strict';

    document.addEventListener('alpine:init', function () {
        Alpine.data('algoCarousel', function () {
            return {
                current: 0,
                count: 4,
                reducedMotion: false,
                scrollTimeout: null,

                init: function () {
                    var self = this;
                    self.count = 4;
                    self.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

                    // 键盘导航（仅在首页聚焦时）
                    window.addEventListener('keydown', function (e) {
                        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
                        var active = document.activeElement;
                        var tag = active ? active.tagName : '';
                        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

                        // 仅当当前 section 为首页时响应
                        var app = window.app || Alpine.store('app');
                        if (app && app.currentSection !== 0) return;

                        e.preventDefault();
                        if (e.key === 'ArrowLeft') self.prev();
                        else self.next();
                    });

                    // 监听 reduced-motion 变化
                    window.matchMedia('(prefers-reduced-motion: reduce)')
                        .addEventListener('change', function (e) {
                            self.reducedMotion = e.matches;
                        });
                },

                get track() {
                    return this.$refs.track;
                },

                onScroll: function () {
                    var self = this;
                    if (this.scrollTimeout) clearTimeout(this.scrollTimeout);
                    this.scrollTimeout = setTimeout(function () {
                        self.syncCurrentFromScroll();
                    }, 80);
                },

                syncCurrentFromScroll: function () {
                    var track = this.track;
                    if (!track) return;
                    var slideWidth = track.scrollWidth / this.count;
                    var idx = Math.round(track.scrollLeft / slideWidth);
                    this.current = Math.max(0, Math.min(this.count - 1, idx));
                },

                goTo: function (idx) {
                    idx = Math.max(0, Math.min(this.count - 1, idx));
                    this.current = idx;
                    var track = this.track;
                    if (!track) return;
                    var slideWidth = track.scrollWidth / this.count;
                    track.scrollTo({
                        left: slideWidth * idx,
                        behavior: this.reducedMotion ? 'auto' : 'smooth'
                    });
                },

                next: function () {
                    this.goTo(Math.min(this.count - 1, this.current + 1));
                },

                prev: function () {
                    this.goTo(Math.max(0, this.current - 1));
                }
            };
        });
    });
})();
