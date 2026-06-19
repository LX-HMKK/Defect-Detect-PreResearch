/**
 * Animation utility library — WAAPI (Web Animations API) 封装
 *
 * 提供统一的入场动画、弹性缩放、子元素逐级延迟等能力。
 * 所有动画返回 Animation 对象，可用于暂停/取消/监听事件。
 *
 * 缓动曲线说明:
 *   --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1)  — 快速衰减，适合入场
 *   --ease-spring:   cubic-bezier(0.22, 0.8, 0.3, 1.15) — 弹性过冲，适合交互反馈
 */
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
        return el.animate(
            [
                { opacity: 0, transform: `translateY(${distance}px)` },
                { opacity: 1, transform: 'translateY(0)' }
            ],
            {
                duration,
                delay,
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
    springScale(el, from = 0.94, to = 1) {
        return el.animate(
            [
                { transform: `scale(${from})` },
                { transform: `scale(${to})` }
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
    staggerChildren(container, selector, options = {}) {
        const { staggerMs = 80, duration = 500, distance = 20 } = options;
        const children = container.querySelectorAll(selector);
        const animations = [];
        children.forEach((child, i) => {
            animations.push(
                this.fadeInUp(child, {
                    delay: i * staggerMs,
                    duration,
                    distance
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
    fadeIn(el, options = {}) {
        const { delay = 0, duration = 400 } = options;
        return el.animate(
            [
                { opacity: 0 },
                { opacity: 1 }
            ],
            {
                duration,
                delay,
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
    fadeOut(el, options = {}) {
        const { delay = 0, duration = 300 } = options;
        return el.animate(
            [
                { opacity: 1 },
                { opacity: 0 }
            ],
            {
                duration,
                delay,
                easing: 'cubic-bezier(0, 0, 0.2, 1)',
                fill: 'both'
            }
        );
    },

    /**
     * 数字滚动效果（从 from 到 to，easeOutExpo 缓动）
     * @param {Element} el - 目标文本元素
     * @param {number} from - 起始值
     * @param {number} to - 结束值
     * @param {number} duration - 持续时间（ms）
     */
    numberRoll(el, from, to, duration = 600) {
        const start = performance.now();

        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // easeOutExpo: 1 - 2^(-10t)
            const eased = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
            const current = from + (to - from) * eased;
            // 根据目标值的小数位数动态调整显示精度
            el.textContent = current.toFixed(4);
            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                el.textContent = to.toFixed(4);
            }
        }
        requestAnimationFrame(update);
    },

    /**
     * 滚动驱动淡入（IntersectionObserver）
     * 对带有 .scroll-reveal 类的元素，当进入视口时触发 fadeInUp。
     * 每个元素仅触发一次（observe → unobserve）。
     */
    initScrollReveal() {
        const elements = document.querySelectorAll('.scroll-reveal');
        if (elements.length === 0) return;

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
    springTo(el, property, from, to, { stiffness = 0.1, damping = 0.5, precision = 0.001 } = {}) {
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
    }
};

/**
 * 全局初始化：在 Alpine 渲染完成后调用，触发滚动淡入动画。
 */
function initAllAnimations() {
    Anim.initScrollReveal();
}
window.initAllAnimations = initAllAnimations;
