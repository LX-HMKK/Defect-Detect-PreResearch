/**
 * Animation utility library — WAAPI (Web Animations API) 封装
 *
 * 提供统一的入场动画、弹性缩放、子元素逐级延迟等能力。
 * 所有动画返回 Animation 对象，可用于暂停/取消/监听事件。
 *
 * 缓动曲线说明:
 *   --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1)  — 快速衰减，适合入场
 *   --ease-spring:   cubic-bezier(0.22, 0.8, 0.3, 1.15) — 弹性过冲，适合交互反馈
 *
 * 无障碍：所有动画函数在 prefers-reduced-motion 时自动跳过，
 *         直接将元素设为目标状态（duration: 0）。
 */

// 无障碍检测：用户是否偏好减少动效
var _prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', function (e) {
    _prefersReducedMotion = e.matches;
});

/**
 * 获取实际动画时长：reduced-motion 时返回 0（瞬间完成）
 */
function _animDuration(ms) {
    return _prefersReducedMotion ? 0 : ms;
}

const Anim = {
    /**
     * 向上淡入
     * @param {Element} el - 目标元素
     * @param {Object} options
     * @param {number} options.delay - 延迟（ms）
     * @param {number} options.duration - 持续时间（ms）
     * @param {number} options.distance - 向上移动距离（px）
     * @returns {Animation}
     */
    fadeInUp(el, options = {}) {
        const { delay = 0, duration = 500, distance = 24 } = options;
        const dur = _animDuration(duration);
        // reduced-motion: 立即显示最终状态，无过渡
        if (dur === 0) {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
            return { cancel: function () {}, finished: Promise.resolve() };
        }
        return el.animate(
            [
                { opacity: 0, transform: 'translateY(' + distance + 'px)' },
                { opacity: 1, transform: 'translateY(0)' }
            ],
            {
                duration: dur,
                delay: _prefersReducedMotion ? 0 : delay,
                easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
                fill: 'both'
            }
        );
    },

    /**
     * 弹性缩放（交互反馈用）
     * @param {Element} el - 目标元素
     * @param {number} from - 起始缩放比例
     * @param {number} to - 结束缩放比例
     * @returns {Animation}
     */
    springScale(el, from, to) {
        from = from !== undefined ? from : 0.94;
        to = to !== undefined ? to : 1;
        if (_prefersReducedMotion) {
            el.style.transform = 'scale(' + to + ')';
            return { cancel: function () {}, finished: Promise.resolve() };
        }
        return el.animate(
            [
                { transform: 'scale(' + from + ')' },
                { transform: 'scale(' + to + ')' }
            ],
            {
                duration: 300,
                easing: 'cubic-bezier(0.22, 0.8, 0.3, 1.15)',
                fill: 'both'
            }
        );
    },

    /**
     * 子元素逐级入场
     * 对容器内匹配的子元素依次应用 fadeInUp，每个元素延迟递增。
     *
     * @param {Element} container - 父容器
     * @param {string} selector - 子元素 CSS 选择器
     * @param {Object} options
     * @param {number} options.staggerMs - 每个子元素之间的延迟增量（ms）
     * @param {number} options.duration - 单个动画持续时间（ms）
     * @param {number} options.distance - 向上移动距离（px）
     * @returns {Animation[]}
     */
    staggerChildren(container, selector, options) {
        options = options || {};
        const { staggerMs = 80, duration = 500, distance = 20 } = options;
        const children = container.querySelectorAll(selector);
        const animations = [];
        children.forEach((child, i) => {
            animations.push(
                this.fadeInUp(child, {
                    delay: i * staggerMs,
                    duration: duration,
                    distance: distance
                })
            );
        });
        return animations;
    },

    /**
     * 淡入（无位移）
     * @param {Element} el
     * @param {Object} options
     * @returns {Animation}
     */
    fadeIn(el, options) {
        options = options || {};
        const { delay = 0, duration = 400 } = options;
        if (_prefersReducedMotion) {
            el.style.opacity = '1';
            return { cancel: function () {}, finished: Promise.resolve() };
        }
        return el.animate(
            [
                { opacity: 0 },
                { opacity: 1 }
            ],
            {
                duration: duration,
                delay: delay,
                easing: 'cubic-bezier(0, 0, 0.2, 1)',
                fill: 'both'
            }
        );
    },

    /**
     * 淡出
     * @param {Element} el
     * @param {Object} options
     * @returns {Animation}
     */
    fadeOut(el, options) {
        options = options || {};
        const { delay = 0, duration = 300 } = options;
        if (_prefersReducedMotion) {
            el.style.opacity = '0';
            return { cancel: function () {}, finished: Promise.resolve() };
        }
        return el.animate(
            [
                { opacity: 1 },
                { opacity: 0 }
            ],
            {
                duration: duration,
                delay: delay,
                easing: 'cubic-bezier(0, 0, 0.2, 1)',
                fill: 'both'
            }
        );
    },

    /**
     * 数字滚动效果（从 from 到 to，弹性过冲缓动）
     *
     * 使用自定义缓动曲线：快速到达目标值后微微超过并回弹，
     * 模拟物理弹簧质感。适合展示得分、置信度等高关注度数字。
     *
     * @param {Element} el - 目标文本元素
     * @param {number} from - 起始值
     * @param {number} to - 结束值
     * @param {number} duration - 持续时间（ms）
     */
    numberRoll(el, from, to, duration, options) {
        duration = duration !== undefined ? duration : 700;
        options = options || {};
        const format = options.format || function (v) { return v.toFixed(4); };

        // reduced-motion: 直接显示最终值
        if (_prefersReducedMotion) {
            el.textContent = format(to);
            return;
        }

        const start = performance.now();

        function elasticEase(t) {
            if (t <= 0) return 0;
            if (t >= 1) return 1;
            const expo = 1 - Math.pow(2, -10 * t);
            if (t < 0.85) return expo;
            const overshoot = (t - 0.85) / 0.15;
            const elastic = Math.sin(overshoot * Math.PI * 2.5) * (1 - overshoot) * 0.03;
            return Math.min(1.03, expo + elastic);
        }

        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = elasticEase(progress);
            const current = from + (to - from) * Math.min(eased, 1.03);
            el.textContent = format(Math.min(current, to * 1.03));
            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                el.textContent = format(to);
            }
        }
        requestAnimationFrame(update);
    },

    /**
     * 滚动驱动淡入（IntersectionObserver）
     * 对带有 .scroll-reveal 类的元素，当进入视口时触发 fadeInUp。
     * 每个元素仅触发一次（observe → unobserve）。
     * reduced-motion 时：直接将所有元素设为可见状态，不创建 Observer。
     */
    initScrollReveal() {
        const elements = document.querySelectorAll('.scroll-reveal');
        if (elements.length === 0) return;

        if (_prefersReducedMotion) {
            // 瞬间显示所有元素
            elements.forEach(function (el) {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            });
            return;
        }

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    Anim.fadeInUp(entry.target, { duration: 500, distance: 24 });
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });

        elements.forEach(el => observer.observe(el));
    },

    /**
     * 弹簧物理动画（模拟弹簧-阻尼系统）
     *
     * @param {Element} el - 目标元素
     * @param {string} property - CSS 属性名
     * @param {number} from - 起始值
     * @param {number|string} to - 目标值（数字默认加 px；字符串直接使用）
     * @param {Object} options
     * @param {number} options.stiffness - 刚度（默认 0.1）
     * @param {number} options.damping - 阻尼（默认 0.5）
     * @param {number} options.precision - 停止精度（默认 0.001）
     * @returns {Promise}
     */
    springTo(el, property, from, to, opts) {
        opts = opts || {};
        const stiffness = opts.stiffness !== undefined ? opts.stiffness : 0.1;
        const damping = opts.damping !== undefined ? opts.damping : 0.5;
        const precision = opts.precision !== undefined ? opts.precision : 0.001;

        if (_prefersReducedMotion) {
            el.style[property] = typeof to === 'number' ? to + 'px' : to.toString();
            return Promise.resolve();
        }

        return new Promise(resolve => {
            let current = from;
            let velocity = 0;
            const target = typeof to === 'number' ? to : parseFloat(to);
            const unit = typeof to === 'number' ? 'px' : '';

            function step() {
                const force = (target - current) * stiffness;
                velocity = (velocity + force) * (1 - damping);
                current += velocity;

                el.style[property] = current + unit;

                if (Math.abs(current - target) > precision || Math.abs(velocity) > precision) {
                    requestAnimationFrame(step);
                } else {
                    el.style[property] = (typeof to === 'number' ? to : to.toString());
                    resolve();
                }
            }
            requestAnimationFrame(step);
        });
    },

    /**
     * Snap 页面过渡编排
     * 当进入新 section 时，触发对应 section 内子元素的逐级入场动画。
     *
     * @param {Element} section - 进入的 section 元素
     * @param {Object} options
     * @param {number} options.staggerMs - 子元素间延迟 (ms)，默认 80
     * @param {number} options.duration - 单个动画时长 (ms)，默认 500
     * @returns {Animation[]}
     */
    snapPageEnter(section, options) {
        options = options || {};
        const { staggerMs = 80, duration = 500 } = options;
        const inner = section.querySelector(':scope > .snap-page-inner');
        if (!inner) return [];

        // 清除该 section 的 exiting 状态
        section.classList.remove('snap-page--exiting');

        // 获取直接子元素（优先 .scroll-reveal，否则所有直接子元素）
        var children = inner.querySelectorAll(':scope > .scroll-reveal');
        if (children.length === 0) {
            children = inner.querySelectorAll(':scope > *');
        }

        if (children.length === 0) return [];

        // 取消该 section 内所有子元素上正在运行的动画（防止与旧退出动画叠加）
        children.forEach(function (child) {
            child.getAnimations().forEach(function (a) { a.cancel(); });
        });

        // reduced-motion: 瞬间显示所有子元素
        if (_prefersReducedMotion) {
            children.forEach(function (child) {
                child.style.opacity = '1';
                child.style.transform = 'none';
            });
            return [];
        }

        const animations = [];
        children.forEach((child, i) => {
            // 跳过隐藏/零高度元素（含 Alpine x-show 控制的）
            if (child.offsetHeight === 0) return;
            if (child.hasAttribute('x-show') && window.getComputedStyle(child).display === 'none') return;
            child.style.opacity = '0';
            child.style.transform = 'translateY(24px)';
            animations.push(
                child.animate(
                    [
                        { opacity: 0, transform: 'translateY(24px)' },
                        { opacity: 1, transform: 'translateY(0)' }
                    ],
                    {
                        duration: duration,
                        delay: i * staggerMs,
                        easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
                        fill: 'forwards'
                    }
                )
            );
        });
        return animations;
    },

    /**
     * Snap 页面离开动画 — 内容向下推出（与滚动方向一致）
     * @param {Element} section - 离开的 section 元素
     * @returns {Animation[]}
     */
    snapPageExit(section) {
        if (_prefersReducedMotion) return [];

        const inner = section.querySelector(':scope > .snap-page-inner');
        if (!inner) return [];

        // 收集所有可见的直接子元素（包括 .scroll-reveal 和 stagger 容器内的）
        const children = [];
        const candidates = inner.querySelectorAll(':scope > .scroll-reveal, :scope > .scroll-reveal-stagger > .scroll-reveal');
        candidates.forEach(function (child) {
            if (child.offsetHeight === 0) return;
            if (child.hasAttribute('x-show') && window.getComputedStyle(child).display === 'none') return;
            children.push(child);
        });
        if (children.length === 0) return [];

        // 取消该 section 内所有正在运行的动画
        children.forEach(function (child) {
            child.getAnimations().forEach(function (a) { a.cancel(); });
        });

        const animations = [];
        children.forEach(function (child, i) {
            animations.push(
                child.animate(
                    [
                        { opacity: 1, transform: 'translateY(0) scale(1)' },
                        { opacity: 0, transform: 'translateY(30px) scale(0.97)' }
                    ],
                    {
                        duration: 350,
                        delay: i * 40,  // 40ms stagger（0s / 0.04s / 0.08s / 0.12s）
                        easing: 'cubic-bezier(0, 0, 0.2, 1)',
                        fill: 'forwards'
                    }
                )
            );
        });
        return animations;
    }
};

/**
 * 全局初始化：在 Alpine 渲染完成后调用，触发滚动淡入动画。
 */
function initAllAnimations() {
    Anim.initScrollReveal();
}
window.initAllAnimations = initAllAnimations;
