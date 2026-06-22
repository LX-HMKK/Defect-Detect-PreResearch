/**
 * 算法卡片横向轮播 — AirPods Pro 风格
 *
 * 功能：
 * - 横向 scroll-snap 滚动，每次吸附一张卡片
 * - 只有被推至最前的中心卡片为完整大小，两侧卡片缩小并弱化
 * - 当前卡片切换到最前时触发其内部流程图动画
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
                flowchartPlayed: {},
                lastFlowchartIndex: -1,
                lastFlowchartTime: 0,

                init: function () {
                    var self = this;
                    self.count = 4;
                    self.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

                    self.$nextTick(function () {
                        self.goTo(0);
                        setTimeout(function () {
                            self._playFlowchartForIndex(0);
                        }, 600);
                    });

                    var track = self.track;
                    if (track) {
                        track.addEventListener('scroll', function () {
                            self._onScroll();
                        }, { passive: true });
                    }

                    window.addEventListener('keydown', function (e) {
                        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
                        var active = document.activeElement;
                        var tag = active ? active.tagName : '';
                        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

                        var app = window.app || Alpine.store('app');
                        if (app && app.currentSection !== 0) return;

                        e.preventDefault();
                        if (e.key === 'ArrowLeft') self.prev();
                        else self.next();
                    });

                    window.matchMedia('(prefers-reduced-motion: reduce)')
                        .addEventListener('change', function (e) {
                            self.reducedMotion = e.matches;
                        });
                },

                get track() {
                    return this.$refs.track;
                },

                _onScroll: function () {
                    var self = this;
                    if (self.scrollTimeout) clearTimeout(self.scrollTimeout);
                    self.scrollTimeout = setTimeout(function () {
                        self.syncCurrentFromScroll();
                    }, 80);
                },

                _slides: function () {
                    var track = this.track;
                    if (!track) return [];
                    return Array.prototype.slice.call(track.children).filter(function (el) {
                        return el.classList.contains('algo-card--slide');
                    });
                },

                syncCurrentFromScroll: function () {
                    var track = this.track;
                    var slides = this._slides();
                    if (!track || slides.length === 0) return;
                    var spacer = slides[0].offsetLeft;
                    var target = track.scrollLeft + spacer;
                    var closest = 0;
                    var minDist = Infinity;
                    slides.forEach(function (s, i) {
                        var dist = Math.abs(s.offsetLeft - target);
                        if (dist < minDist) {
                            minDist = dist;
                            closest = i;
                        }
                    });

                    if (closest !== this.current) {
                        this.current = closest;
                        this._updateActiveClasses();
                        this._playFlowchartForIndex(this.current);
                    }
                },

                _updateActiveClasses: function () {
                    var slides = this._slides();
                    slides.forEach(function (slide, i) {
                        slide.classList.toggle('is-active', i === this.current);
                    }, this);
                },

                _playFlowchartForIndex: function (idx) {
                    var now = Date.now();
                    if (idx === this.lastFlowchartIndex && now - this.lastFlowchartTime < 3000) {
                        return;
                    }
                    this.lastFlowchartIndex = idx;
                    this.lastFlowchartTime = now;

                    var slides = this._slides();
                    if (!slides[idx]) return;
                    var svg = slides[idx].querySelector('.flowchart-svg');
                    if (svg && window.animateFlowchart) {
                        window.animateFlowchart(svg);
                    }
                },

                goTo: function (idx) {
                    idx = Math.max(0, Math.min(this.count - 1, idx));
                    this.current = idx;
                    var track = this.track;
                    var slides = this._slides();
                    if (!track || slides.length === 0 || !slides[idx]) return;
                    var spacer = slides[0].offsetLeft;
                    track.scrollTo({
                        left: slides[idx].offsetLeft - spacer,
                        behavior: this.reducedMotion ? 'auto' : 'smooth'
                    });
                    this._updateActiveClasses();
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
