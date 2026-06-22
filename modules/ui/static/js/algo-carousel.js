/**
 * 算法卡片横向轮播 — AirPods Pro 风格
 *
 * 功能：
 * - 横向 scroll-snap 滚动，每次吸附一张卡片
 * - 滚动过程中根据卡片与视口中心的距离实时计算缩放 / Z轴推移 / 模糊，
 *   形成「卡片被推远 / 推到最前」的纵深动效
 * - 当前卡片推到最前时触发其内部流程图动画
 * - 指示点同步、01/04 计数器
 * - 键盘 ←→ 导航、按钮前后切换
 * - 无障碍：prefers-reduced-motion 时禁用滚动动画与连续 transform
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
                rafId: null,
                flowchartPlayed: {},
                lastFlowchartIndex: -1,
                lastFlowchartTime: 0,

                init: function () {
                    var self = this;
                    self.count = 4;
                    self.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

                    // 初始化时将第一张卡片滚动到居中位置，避免首屏左对齐
                    self.$nextTick(function () {
                        self.goTo(0);
                        // 首次布局稳定后触发第一张卡片的流程图
                        setTimeout(function () {
                            self._playFlowchartForIndex(0);
                        }, 600);
                    });

                    // 滚动时实时更新卡片纵深，并在停止后同步当前索引
                    var track = self.track;
                    if (track) {
                        track.addEventListener('scroll', function () {
                            self._onScroll();
                        }, { passive: true });
                    }

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
                            if (self.reducedMotion) {
                                self._resetCardTransforms();
                            } else {
                                self._scheduleUpdate();
                            }
                        });
                },

                get track() {
                    return this.$refs.track;
                },

                _onScroll: function () {
                    var self = this;
                    self._scheduleUpdate();

                    if (self.scrollTimeout) clearTimeout(self.scrollTimeout);
                    self.scrollTimeout = setTimeout(function () {
                        self.syncCurrentFromScroll();
                    }, 80);
                },

                _rafPending: false,
                _scheduleUpdate: function () {
                    if (this._rafPending || this.reducedMotion) return;
                    this._rafPending = true;
                    var self = this;
                    this.rafId = requestAnimationFrame(function () {
                        self._rafPending = false;
                        self._updateCardTransforms();
                    });
                },

                _slides: function () {
                    var track = this.track;
                    if (!track) return [];
                    return Array.prototype.slice.call(track.children).filter(function (el) {
                        return el.classList.contains('algo-card--slide');
                    });
                },

                _updateCardTransforms: function () {
                    var track = this.track;
                    var slides = this._slides();
                    if (!track || slides.length === 0) return;

                    var trackCenter = track.clientWidth / 2;
                    var maxDist = track.clientWidth * 0.6;

                    slides.forEach(function (slide) {
                        var slideCenter = slide.offsetLeft - track.scrollLeft + slide.offsetWidth / 2;
                        var dist = Math.abs(slideCenter - trackCenter);
                        var t = Math.min(1, Math.max(0, dist / maxDist));

                        // 中心卡片推到最前（scale 1, translateZ 40px），两侧推远并弱化
                        var scale = 1 - t * 0.16;
                        var z = 40 - t * 140;
                        var opacity = 1 - t * 0.55;
                        var blur = t * 2.5;

                        slide.style.transform = 'scale(' + scale.toFixed(3) + ') translateZ(' + z.toFixed(1) + 'px)';
                        slide.style.opacity = opacity.toFixed(3);
                        slide.style.filter = 'blur(' + blur.toFixed(2) + 'px)';
                    });
                },

                _resetCardTransforms: function () {
                    var slides = this._slides();
                    slides.forEach(function (slide, i) {
                        slide.style.transform = '';
                        slide.style.opacity = '';
                        slide.style.filter = '';
                        slide.classList.toggle('is-active', i === this.current);
                    }, this);
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
                        var prev = this.current;
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
                    // 同一卡片在 3 秒内不重复播放；快速滑动时只播放最终停留的卡片
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
