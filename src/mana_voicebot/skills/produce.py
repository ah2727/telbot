from __future__ import annotations

from typing import Dict, Any

from openai import OpenAI

from ..core.state import ConversationState, SkillResult
from .base import BaseSkill


class ProduceSkill(BaseSkill):
    """
    Skill 'produce' – تولید متن فروش برای یک محصول (فعلاً TeleBot AI)
    با تمرکز روی سؤال‌هایی مثل «چرا باید این AI را بخرم؟»
    """

    name = "produce"

    def __init__(self, client: OpenAI):
        super().__init__(client)

    def handle(
        self,
        turn_text: str,
        state: ConversationState,
        raw_json: Dict[str, Any],
    ) -> SkillResult:
        intent = str(raw_json.get("intent") or "pitch")

        # ورودی‌ها یا default
        product_name = (raw_json.get("product_name") or "TeleBot AI").strip()
        audience = (raw_json.get("audience") or "کلینیک‌ها، مطب‌ها و کسب‌وکارهای خدماتی").strip()
        pain_point = (
            raw_json.get("pain_point")
            or "اتلاف وقت زیاد روی پاسخ‌گویی تکراری در واتساپ و تلفن و جا افتادن نوبت‌ها"
        ).strip()
        benefit = (
            raw_json.get("benefit")
            or "خودکارسازی پاسخ‌گویی و نوبت‌گیری ۲۴ ساعته، بدون خستگی و بدون خطای انسانی"
        ).strip()
        cta = (
            raw_json.get("cta")
            or "اگر دوست دارید، می‌توانیم یک دمو کوتاه برای شما فعال کنیم تا از نزدیک ببینید چطور کار می‌کند."
        ).strip()

        reply = str(raw_json.get("reply") or "").strip()

        # اگر مغز مرکزی از قبل reply ساخته، همون رو بده
        if reply:
            payload = {
                "intent": intent,
                "product_name": product_name,
                "audience": audience,
                "pain_point": pain_point,
                "benefit": benefit,
                "cta": cta,
            }
            return SkillResult(
                reply=reply,
                domain=self.name,
                intent=intent,
                payload=payload,
            )

        # اگر intent خاص "why_buy" بود → جواب هدفمند "چرا بخرم؟"
        if intent == "why_buy":
            reply = self._build_why_buy_reply(
                product_name=product_name,
                audience=audience,
                pain_point=pain_point,
                benefit=benefit,
                cta=cta,
            )
        else:
            # حالت عمومی pitch
            reply = self._build_generic_pitch(
                product_name=product_name,
                audience=audience,
                pain_point=pain_point,
                benefit=benefit,
                cta=cta,
            )

        payload = {
            "intent": intent,
            "product_name": product_name,
            "audience": audience,
            "pain_point": pain_point,
            "benefit": benefit,
            "cta": cta,
        }

        return SkillResult(
            reply=reply,
            domain=self.name,
            intent=intent,
            payload=payload,
        )

    # ---------- builders ----------

    def _build_why_buy_reply(
        self,
        product_name: str,
        audience: str,
        pain_point: str,
        benefit: str,
        cta: str,
    ) -> str:
        """
        جواب مخصوص سؤال «چرا باید این AI را بخرم؟»
        لحن: دوستانه، محترمانه، بدون اغراق دروغ.
        """

        parts: list[str] = []

        # 1) توضیح ساده چی هست
        parts.append(
            f"{product_name} در واقع یک دستیار هوشمند است که برای {audience} طراحی شده "
            "تا کارهای تکراری مثل نوبت‌گیری و پاسخ‌گویی به سوالات ساده را به‌جای شما انجام بدهد."
        )

        # 2) ربط به مشکل واقعی
        parts.append(
            f"اگر بخواهم ساده بگویم، مشکل اصلی که حل می‌کند این است: «{pain_point}»."
        )

        # 3) چرا ارزش خرید دارد (benefit + زمان + تمرکز)
        parts.append(
            f"با {product_name} این روند {benefit}؛ یعنی به‌جای این‌که منشی یا خودتان مدام پای گوشی و واتساپ باشید، "
            "زمان‌تان آزاد می‌شود و می‌توانید روی کارهای تخصصی‌تر تمرکز کنید."
        )

        # 4) کاهش ریسک / لحن مطمئن اما منطقی
        parts.append(
            "نکته مهم این است که این سیستم قرار نیست جای شما را بگیرد؛ "
            "فقط کارهای خسته‌کننده و تکراری را از روی دوش‌تان برمی‌دارد و خطاهای انسانی را کمتر می‌کند."
        )

        # 5) CTA نرم
        parts.append(
            cta
        )

        return " ".join(parts)

    def _build_generic_pitch(
        self,
        product_name: str,
        audience: str,
        pain_point: str,
        benefit: str,
        cta: str,
    ) -> str:
        """
        جواب عمومی معرفی محصول، اگر intent غیر از why_buy بود.
        """
        parts: list[str] = []

        parts.append(
            f"{product_name} برای {audience} طراحی شده تا دردسر مدیریت نوبت و پاسخ‌گویی تکراری را کم کند."
        )
        parts.append(
            f"خیلی از کسب‌وکارها با مشکل «{pain_point}» درگیر هستند و این هم زمان می‌گیرد هم روی کیفیت کار اثر می‌گذارد."
        )
        parts.append(
            f"{product_name} این روند را خودکار می‌کند؛ یعنی {benefit} و شما می‌توانید روی کارهای مهم‌تر تمرکز کنید."
        )
        parts.append(
            "تجربه نشان داده وقتی پاسخ‌گویی سریع و منظم شود، رضایت مراجعین هم بالاتر می‌رود و تماس‌های عصبی کمتر می‌شود."
        )
        parts.append(cta)

        return " ".join(parts)
