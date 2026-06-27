/**
 * 算法卡片横向轮播 — AirPods Pro 风格"灵动滚动条"
 *
 * 与旧版的差异（核心改造）：
 * - 连续深度：不再用 is-active 在 scale(0.72/1) 间硬切，而是每帧根据每张卡片
 *   距轨道中心的距离计算 --card-closeness（0=远离, 1=居中），由 CSS 连续驱动
 *   scale / translateZ / opacity / blur / brightness / saturate / box-shadow。
 * - 流程图生命周期：卡片推到最前 → reset + animate（开始渲染）；退出最前 →
 *   reset（关闭渲染）；再次推到最前 → 重新从头渲染（去掉了阻止重播的 3s 去抖）。
 * - 进度条：随横向滚动连续填充的 .algo-carousel-progress-fill，呼应"灵动滚动条"。
 * - 活动算法色 tint：当前卡片颜色提升为 --active-algo，驱动进度条与卡片辉光。
 * - 视口生命周期：IntersectionObserver 监听轮播是否在 .snap-container 可见，
 *   离开页面时关闭渲染，返回时重新渲染。
 *
 * 无障碍：prefers-reduced-motion 时改为离散 0/1 closeness，并禁用滚动动画。
 */
(function() {
    'use strict';

    document.addEventListener('alpine:init', function () {
        Alpine.data('algoCarousel', function () {
            return {
                current: 0,
                count: 4,
                reducedMotion: false,
                _layoutScheduled: false,
                _playTimer: null,
                _io: null,

                init: function () {
                    var self = this;
                    self.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

                    self.$nextTick(function () {
                        var slides = self._slides();
                        self.count = slides.length || 4;
                        self._updateActiveClasses();
                        self._setActiveTint();
                        self._layout();
                        // 首屏：居中卡片渲染流程图
                        self._schedulePlay(self.current);
                    });

                    var track = self.track;
                    if (track) {
                        track.addEventListener('scroll', function () {
                            self._requestLayout();
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
                            self._requestLayout();
                        });

                    window.addEventListener('resize', function () {
                        self._requestLayout();
                    }, { passive: true });

                    // 视口生命周期：离开页面 → 关闭渲染；返回 → 重新渲染
                    self._setupVisibility();
                },

                get track() {
                    return this.$refs.track;
                },

                get progressFill() {
                    return this.$refs.carouselProgressFill;
                },

                _slides: function () {
                    var track = this.track;
                    if (!track) return [];
                    return Array.prototype.slice.call(track.children).filter(function (el) {
                        return el.classList.contains('algo-card--slide');
                    });
                },

                /* ── 滚动驱动：每帧更新 closeness + 进度 + 活动卡片 ── */
                _requestLayout: function () {
                    var self = this;
                    if (self._layoutScheduled) return;
                    self._layoutScheduled = true;
                    requestAnimationFrame(function () {
                        self._layoutScheduled = false;
                        self._layout();
                    });
                },

                _layout: function () {
                    var track = this.track;
                    var slides = this._slides();
                    if (!track || slides.length === 0) return;

                    var center = track.scrollLeft + track.clientWidth / 2;
                    var maxScroll = track.scrollWidth - track.clientWidth;
                    var progress = maxScroll > 0 ? track.scrollLeft / maxScroll : 0;

                    var closest = 0;
                    var minDist = Infinity;

                    slides.forEach(function (slide, i) {
                        var slideCenter = slide.offsetLeft + slide.offsetWidth / 2;
                        var dist = Math.abs(slideCenter - center);
                        var cardW = slide.offsetWidth || 1;

                        var closeness;
                        if (this.reducedMotion) {
                            // 离散：当前卡 1，其余 0
                            closeness = (i === this.current) ? 1 : 0;
                        } else {
                            // 连续：距中心越近越接近 1，并用 smoothstep 平滑
                            var c = 1 - Math.min(dist / cardW, 1);
                            closeness = c * c * (3 - 2 * c);
                        }
                        slide.style.setProperty('--card-closeness', closeness.toFixed(4));

                        if (dist < minDist) {
                            minDist = dist;
                            closest = i;
                        }
                    }, this);

                    // 进度条
                    if (this.progressFill) {
                        var pct = Math.max(0, Math.min(1, progress)) * 100;
                        this.progressFill.style.width = pct.toFixed(2) + '%';
                    }

                    // 活动卡片切换 → 触发流程图生命周期
                    if (closest !== this.current) {
                        var prev = this.current;
                        this.current = closest;
                        this._updateActiveClasses();
                        this._setActiveTint();
                        this._stopPlay(prev);       // 退出最前：关闭渲染
                        this._schedulePlay(closest); // 推到最前：开始渲染（带 dwell）
                    }
                },

                _updateActiveClasses: function () {
                    var slides = this._slides();
                    slides.forEach(function (slide, i) {
                        slide.classList.toggle('is-active', i === this.current);
                    }, this);
                },

                _setActiveTint: function () {
                    var root = this.$el;
                    var slides = this._slides();
                    if (!root || !slides[this.current]) return;
                    var color = slides[this.current].style.getPropertyValue('--algo-color');
                    if (!color) {
                        // 回退到 computed style（兼容内联写在子元素的情况）
                        color = getComputedStyle(slides[this.current]).getPropertyValue('--algo-color');
                    }
                    color = (color || '').trim() || '#2997ff';
                    root.style.setProperty('--active-algo', color);
                },

                /* ── 流程图生命周期 ── */
                _schedulePlay: function (idx) {
                    var self = this;
                    if (self._playTimer) clearTimeout(self._playTimer);
                    self._playTimer = setTimeout(function () {
                        self._playTimer = null;
                        self._playFlowchart(idx);
                    }, 170); // dwell：仅在卡片稳定居中后才渲染，避免快速掠过时闪烁
                },

                _stopPlay: function (idx) {
                    if (this._playTimer) {
                        clearTimeout(this._playTimer);
                        this._playTimer = null;
                    }
                    var slides = this._slides();
                    if (idx == null || !slides[idx]) return;
                    var svg = slides[idx].querySelector('.flowchart-svg');
                    if (svg && window.resetFlowchart) {
                        window.resetFlowchart(svg); // 关闭渲染：复位到未绘制状态
                    }
                },

                _playFlowchart: function (idx) {
                    if (idx !== this.current) return; // 已不再是当前卡
                    var slides = this._slides();
                    if (!slides[idx]) return;
                    var svg = slides[idx].querySelector('.flowchart-svg');
                    if (!svg) return;
                    if (window.resetFlowchart) window.resetFlowchart(svg);   // 确保从头开始
                    if (window.animateFlowchart) window.animateFlowchart(svg); // 开始渲染
                },

                /* ── 视口生命周期：离开页面关闭渲染，返回重新渲染 ── */
                _setupVisibility: function () {
                    var self = this;
                    var rootEl = document.querySelector('.snap-container');
                    if (!rootEl || !self.$el) return;
                    self._io = new IntersectionObserver(function (entries) {
                        entries.forEach(function (entry) {
                            if (entry.isIntersecting) {
                                self._schedulePlay(self.current);
                            } else {
                                self._stopPlay(self.current);
                            }
                        });
                    }, { root: rootEl, threshold: 0.35 });
                    self._io.observe(self.$el);
                },

                /* ── 导航 ── */
                goTo: function (idx) {
                    idx = Math.max(0, Math.min(this.count - 1, idx));
                    var track = this.track;
                    var slides = this._slides();
                    if (!track || slides.length === 0 || !slides[idx]) return;
                    var spacer = slides[0].offsetLeft;
                    track.scrollTo({
                        left: slides[idx].offsetLeft - spacer,
                        behavior: this.reducedMotion ? 'auto' : 'smooth'
                    });
                    // current / 活动类 / tint / 流程图 由滚动驱动的 _layout 统一更新，
                    // 避免与滚动位置争抢状态。
                    this._requestLayout();
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
