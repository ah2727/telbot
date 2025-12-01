from __future__ import annotations

from typing import Dict, Any

from openai import OpenAI

from ..core.state import ConversationState, SkillResult
from .base import BaseSkill


class VisitorSkill(BaseSkill):
    """
    Skill 'visitor' – اپراتور تلفنی برای معرفی و فروش TeleBot AI.
    شروع مکالمه، پرسیدن نیاز، توضیح محصول، پاسخ به سؤال‌ها و جمع‌بندی برای اقدام بعدی.
    """

    name = "visitor"

    def __init__(self, client: OpenAI):
        super().__init__(client)

    def handle(
        self,
        turn_text: str,
        state: ConversationState,
        raw_json: Dict[str, Any],
    ) -> SkillResult:
        # intent کلی از مغز مرکزی
        intent = str(raw_json.get("intent") or "intro")

        # فیلدهایی که مغز مرکزی می‌تواند پر کند
        product_name = (raw_json.get("product_name") or "TeleBot AI").strip()
        visitor_name = (raw_json.get("visitor_name") or "").strip()
        business_type = (raw_json.get("business_type") or "").strip()
        main_question = (raw_json.get("question") or "").strip()
        kb_answer = (raw_json.get("kb_answer") or "").strip()  # جواب از سرچ/دانش‌پایه، اگر بود
        reply = str(raw_json.get("reply") or "").strip()

        # اگر name از اینجا آمد، در پروفایل نگه دار
        if visitor_name:
            state.profile["name"] = visitor_name

        # state مخصوص visitor را آپدیت کن
        vs = state.visitor_state
        vs["last_intent"] = intent
        if visitor_name:
            vs["visitor_name"] = visitor_name
        if business_type:
            vs["business_type"] = business_type
        if main_question:
            vs["last_question"] = main_question

        # اگر مغز مرکزی خودش reply ساخته، همون را استفاده کن
        if reply:
            vs["last_reply"] = reply
            payload = {
                "intent": intent,
                "product_name": product_name,
                "visitor_name": visitor_name or state.profile.get("name"),
                "business_type": business_type or vs.get("business_type"),
                "question": main_question,
            }
            return SkillResult(
                reply=reply,
                domain=self.name,
                intent=intent,
                payload=payload,
            )

        # اگر kb_answer داریم، در جواب استفاده‌اش کن
        if kb_answer:
            reply = self._build_kb_based_reply(
                product_name=product_name,
                kb_answer=kb_answer,
            )
        else:
            # بر اساس intent یک جواب پیش‌فرض بساز
            if intent == "intro":
                reply = self._build_intro(product_name, visitor_name)
            elif intent == "needs":
                reply = self._build_needs_question(product_name, visitor_name)
            elif intent == "product_info":
                reply = self._build_product_info(product_name)
            elif intent == "pricing":
                reply = self._build_pricing(product_name)
            elif intent == "objection":
                reply = self._build_objection_handle(product_name, main_question)
            elif intent == "closing":
                reply = self._build_closing(product_name)
            else:
                # fallback
                reply = (
                    f"من دستیار فروش {product_name} هستم. "
                    "می‌خواهید ابتدا درباره امکاناتش بگویم یا درباره قیمت و نحوه راه‌اندازی بپرسید؟"
                )

        vs["last_reply"] = reply

        payload = {
            "intent": intent,
            "product_name": product_name,
            "visitor_name": visitor_name or state.profile.get("name"),
            "business_type": business_type or vs.get("business_type"),
            "question": main_question,
        }

        return SkillResult(
            reply=reply,
            domain=self.name,
            intent=intent,
            payload=payload,
        )

    # ---------- builders ----------

    def _build_intro(self, product_name: str, visitor_name: str | None) -> str:
        if visitor_name:
            name_part = f"{visitor_name} عزیز، "
        else:
            name_part = ""

        return (
            f"{name_part}سلام، من دستیار تلفنی {product_name} هستم. "
            "به شما کمک می‌کنم ببینید این ربات دقیقاً چه کمکی به کسب‌وکار شما می‌کند. "
            "اول بفرمایید در چه نوع کسب‌وکاری فعال هستید و بیشتر چه کانالی برای نوبت‌گیری یا چت با مشتری دارید؛ تلفن، واتساپ یا شبکه‌های اجتماعی؟"
        )

    def _build_needs_question(self, product_name: str, visitor_name: str | None) -> str:
        prefix = "خیلی خوب. " if visitor_name else ""
        return (
            f"{prefix}برای این‌که بهتر راهنمایی‌تان کنم، بفرمایید الان بزرگ‌ترین مشکل‌تان در ارتباط با مراجعین چیست؟ "
            "مثلاً حجم زیاد پیام‌ها، جا افتادن نوبت، پاسخ‌گویی خارج از ساعت کاری یا چیز دیگری؟"
        )

    def _build_product_info(self, product_name: str) -> str:
        return (
            f"{product_name} در عمل یک اپراتور هوشمند است که روی واتساپ یا کانال‌های دیگر شما می‌نشیند؛ "
            "سوالات تکراری را جواب می‌دهد، نوبت‌گیری را خودکار می‌کند و مکالمات را برای شما ثبت می‌کند. "
            "مزیتش این است که ۲۴ ساعته فعال است، خسته نمی‌شود و می‌تواند هم‌زمان با چند نفر صحبت کند، "
            "بدون این‌که نیاز باشد شما یا منشی همیشه پای گوشی باشید."
        )

    def _build_pricing(self, product_name: str) -> str:
        # بدون عدد دقیق؛ فقط چارچوب امن
        return (
            f"قیمت {product_name} معمولاً به تعداد کانال‌ها، حجم استفاده و امکاناتی که نیاز دارید بستگی دارد. "
            "ما معمولاً یک دورهٔ آزمایشی یا پلن شروع سبک پیشنهاد می‌کنیم تا ببینید دقیقاً چقدر به‌دردتان می‌خورد، "
            "بعد می‌شود درباره پلن نهایی تصمیم گرفت. اگر بفرمایید حدوداً روزانه چند پیام یا تماس دارید، "
            "می‌توانم بهتر راهنمایی کنم چه مدلی مناسب شماست."
        )

    def _build_objection_handle(self, product_name: str, question: str) -> str:
        # جواب عمومی برای نگرانی‌ها؛ خود مغز مرکزی می‌تواند این intent را به‌همراه kb_answer پر کند
        base = (
            f"نگرانی‌تان کاملاً قابل درک است. هدف {product_name} جایگزین‌کردن انسان نیست، "
            "بلکه گرفتن کارهای تکراری و خسته‌کننده از روی دوش شما و تیمتان است."
        )
        if not question:
            return base + " اگر دوست دارید بفرمایید دقیقاً از چه چیزی نگران هستید تا همان را شفاف توضیح بدهم."
        return (
            base
            + " "
            + "اگر دقیق‌تر بفرمایید در مورد چه موضوعی تردید دارید "
            "— مثلاً امنیت، کیفیت پاسخ‌گویی یا هزینه — می‌توانم همان بخش را شفاف‌تر توضیح بدهم."
        )

    def _build_closing(self, product_name: str) -> str:
        return (
            f"خلاصه اگر بخواهم بگویم، {product_name} کمک می‌کند مدیریت پیام‌ها و نوبت‌گیری شما منظم و خودکار شود "
            "و وقت آزاد بیشتری برای کارهای مهم‌تر داشته باشید. "
            "اگر مایل باشید، می‌توانیم از یک دمو یا دورهٔ کوتاه شروع کنیم تا در عمل ببینید چقدر برای شما مناسب است."
        )

    def _build_kb_based_reply(self, product_name: str, kb_answer: str) -> str:
        """
        وقتی از بیرون (سرچ/دانش‌پایه) جواب آماده شده در kb_answer می‌آید،
        این‌جا آن را به یک پاسخ گفت‌وگویی تبدیل می‌کنیم.
        """
        return (
            f"در مورد {product_name} این‌طور می‌توانم توضیح بدهم: {kb_answer} "
            "اگر چیزی هنوز مبهم است، لطفاً دقیق‌تر بپرسید تا همان قسمت را روشن‌تر کنم."
        )
