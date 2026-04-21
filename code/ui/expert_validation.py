"""
Expert Validation Interface
Allows Lebanese lawyers to rate and validate AI-generated legal answers
Critical for thesis validation study
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from loguru import logger


class ExpertValidationUI:
    """
    Interface for expert validation of legal AI responses.

    Addresses Thesis Validation:
    - Collect expert ratings on answer quality
    - Track inter-rater reliability
    - Provide structured feedback for improvement
    """

    def __init__(self, results_dir: str = "./data_processed/validation"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.validation_file = self.results_dir / "expert_ratings.json"
        self.ratings = self._load_ratings()

    def _load_ratings(self) -> List[Dict]:
        """Load existing expert ratings."""
        if self.validation_file.exists():
            try:
                with open(self.validation_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_rating(self, rating: Dict):
        """Save a new expert rating."""
        self.ratings.append(rating)

        with open(self.validation_file, "w", encoding="utf-8") as f:
            json.dump(self.ratings, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved expert rating for query: {rating['query_id']}")

    def render_validation_interface(self, query: str, answer: str,
                                   citations: List[str], query_id: str = None):
        """
        Render expert validation interface.

        Args:
            query: Original legal query
            answer: AI-generated answer
            citations: List of cited articles
            query_id: Unique query identifier
        """

        if query_id is None:
            query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        st.markdown("---")
        st.subheader("👨‍⚖️ Expert Validation")

        st.info("""
        **للمحامين والخبراء القانونيين**

        يرجى تقييم الإجابة المُقدمة من النظام وفقاً للمعايير التالية.
        تقييمك سيساعد في تحسين النظام وتوثيق فعاليته في البحث العلمي.
        """)

        # Expert Information
        with st.expander("معلومات الخبير / Expert Information", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                expert_id = st.text_input(
                    "رقم الخبير (اختياري) / Expert ID",
                    help="معرّف مجهول للخبير"
                )

                years_experience = st.selectbox(
                    "سنوات الخبرة / Years of Experience",
                    ["<5", "5-10", "10-20", "20+"]
                )

            with col2:
                specialization = st.multiselect(
                    "التخصص القانوني / Legal Specialization",
                    [
                        "Civil Law / القانون المدني",
                        "Contract Law / قانون العقود",
                        "Criminal Law / القانون الجزائي",
                        "Commercial Law / القانون التجاري",
                        "Labor Law / قانون العمل",
                        "Other / أخرى"
                    ]
                )

                language_preference = st.radio(
                    "لغة التقييم / Rating Language",
                    ["العربية", "Français", "English"],
                    horizontal=True
                )

        # Rating Criteria
        st.markdown("### معايير التقييم / Rating Criteria")

        criteria = {}

        # Criterion 1: Legal Accuracy
        st.markdown("**1. الدقة القانونية / Legal Accuracy**")
        criteria['legal_accuracy'] = st.slider(
            "هل الإجابة دقيقة قانونياً؟ / Is the answer legally accurate?",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = خاطئة تماماً / Completely incorrect\n5 = دقيقة تماماً / Completely accurate"
        )

        # Criterion 2: Completeness
        st.markdown("**2. الشمولية / Completeness**")
        criteria['completeness'] = st.slider(
            "هل تغطي الإجابة جميع جوانب السؤال؟ / Does it cover all aspects?",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = ناقصة جداً / Very incomplete\n5 = شاملة تماماً / Completely comprehensive"
        )

        # Criterion 3: Citation Quality
        st.markdown("**3. جودة الاستشهادات / Citation Quality**")
        criteria['citation_quality'] = st.slider(
            "هل الاستشهادات القانونية مناسبة ودقيقة؟ / Are citations appropriate?",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = غير مناسبة / Not appropriate\n5 = مناسبة تماماً / Perfectly appropriate"
        )

        # Criterion 4: Clarity
        st.markdown("**4. الوضوح / Clarity**")
        criteria['clarity'] = st.slider(
            "هل الإجابة واضحة وسهلة الفهم؟ / Is it clear and understandable?",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = غير واضحة / Not clear\n5 = واضحة جداً / Very clear"
        )

        # Criterion 5: Practical Utility
        st.markdown("**5. الفائدة العملية / Practical Utility**")
        criteria['practical_utility'] = st.slider(
            "هل يمكن استخدام هذه الإجابة في الممارسة القانونية؟ / Useful in practice?",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = غير مفيدة / Not useful\n5 = مفيدة جداً / Very useful"
        )

        # Overall Rating
        st.markdown("### التقييم الإجمالي / Overall Rating")
        overall_rating = st.slider(
            "التقييم الإجمالي للإجابة / Overall answer quality",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = ضعيف جداً / Very poor\n5 = ممتاز / Excellent"
        )

        # Would Use in Practice?
        would_use = st.radio(
            "هل ستستخدم هذا النظام في عملك القانوني؟ / Would you use this system?",
            ["نعم / Yes", "ربما / Maybe", "لا / No"],
            horizontal=True
        )

        # Qualitative Feedback
        st.markdown("### ملاحظات إضافية / Additional Comments")

        col1, col2 = st.columns(2)

        with col1:
            strengths = st.text_area(
                "نقاط القوة / Strengths",
                help="ما هي نقاط القوة في الإجابة؟"
            )

        with col2:
            weaknesses = st.text_area(
                "نقاط الضعف / Weaknesses",
                help="ما هي نقاط الضعف أو المشاكل؟"
            )

        suggestions = st.text_area(
            "اقتراحات للتحسين / Suggestions for Improvement",
            help="كيف يمكن تحسين الإجابة؟"
        )

        # Errors Identified
        st.markdown("### الأخطاء المحددة / Identified Errors")

        has_errors = st.checkbox("يوجد أخطاء قانونية / Contains legal errors")

        errors = []
        if has_errors:
            error_type = st.multiselect(
                "نوع الخطأ / Error Type",
                [
                    "استشهاد خاطئ / Incorrect citation",
                    "تفسير خاطئ للقانون / Misinterpretation of law",
                    "معلومات قديمة / Outdated information",
                    "معلومات ناقصة / Missing information",
                    "تطبيق خاطئ / Incorrect application",
                    "أخرى / Other"
                ]
            )

            error_description = st.text_area(
                "وصف الأخطاء / Error Description",
                help="يرجى وصف الأخطاء بالتفصيل"
            )

            errors = {
                "has_errors": True,
                "error_types": error_type,
                "description": error_description
            }

        # Submit Rating
        col1, col2, col3 = st.columns([2, 1, 2])

        with col2:
            submit = st.button("✅ إرسال التقييم / Submit Rating", use_container_width=True)

        if submit:
            # Validate required fields
            if not expert_id:
                st.warning("⚠️ يرجى إدخال رقم الخبير / Please enter Expert ID")
                return

            # Calculate average score
            avg_criteria = sum(criteria.values()) / len(criteria)

            # Build rating object
            rating = {
                "query_id": query_id,
                "query": query,
                "answer": answer,
                "citations": citations,
                "timestamp": datetime.now().isoformat(),
                "expert": {
                    "id": expert_id,
                    "years_experience": years_experience,
                    "specialization": specialization,
                    "language": language_preference
                },
                "ratings": {
                    **criteria,
                    "overall": overall_rating,
                    "average_criteria": round(avg_criteria, 2)
                },
                "would_use": would_use,
                "feedback": {
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "suggestions": suggestions
                },
                "errors": errors if has_errors else {"has_errors": False}
            }

            # Save rating
            self._save_rating(rating)

            st.success("""
            ✅ **تم إرسال التقييم بنجاح!**

            شكراً لمساهمتك في تحسين النظام.
            تقييمك سيُستخدم في البحث العلمي لتوثيق فعالية الذكاء الاصطناعي في المجال القانوني اللبناني.
            """)

            # Show summary
            with st.expander("ملخص التقييم / Rating Summary"):
                st.json(rating)

    def render_validation_dashboard(self):
        """Render dashboard showing all expert validations."""

        st.title("📊 Expert Validation Dashboard")

        if not self.ratings:
            st.info("لا توجد تقييمات حتى الآن / No ratings yet")
            return

        # Summary Statistics
        st.subheader("📈 إحصائيات عامة / Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("عدد التقييمات / Total Ratings", len(self.ratings))

        with col2:
            avg_overall = sum(r['ratings']['overall'] for r in self.ratings) / len(self.ratings)
            st.metric("التقييم الإجمالي / Avg Overall", f"{avg_overall:.2f}/5")

        with col3:
            num_experts = len(set(r['expert']['id'] for r in self.ratings))
            st.metric("عدد الخبراء / Experts", num_experts)

        with col4:
            with_errors = sum(1 for r in self.ratings if r['errors']['has_errors'])
            error_rate = (with_errors / len(self.ratings)) * 100
            st.metric("نسبة الأخطاء / Error Rate", f"{error_rate:.1f}%")

        # Detailed Ratings Table
        st.subheader("📋 جميع التقييمات / All Ratings")

        # Convert to DataFrame
        ratings_data = []
        for r in self.ratings:
            ratings_data.append({
                "Query ID": r['query_id'],
                "Query": r['query'][:50] + "...",
                "Expert": r['expert']['id'],
                "Experience": r['expert']['years_experience'],
                "Legal Accuracy": r['ratings']['legal_accuracy'],
                "Completeness": r['ratings']['completeness'],
                "Citation Quality": r['ratings']['citation_quality'],
                "Clarity": r['ratings']['clarity'],
                "Practical Utility": r['ratings']['practical_utility'],
                "Overall": r['ratings']['overall'],
                "Would Use": r['would_use'],
                "Has Errors": "✗" if r['errors']['has_errors'] else "✓",
                "Date": r['timestamp'][:10]
            })

        df = pd.DataFrame(ratings_data)
        st.dataframe(df, use_container_width=True)

        # Criteria Breakdown
        st.subheader("📊 تفصيل المعايير / Criteria Breakdown")

        criteria_names = [
            'legal_accuracy', 'completeness', 'citation_quality',
            'clarity', 'practical_utility'
        ]

        criteria_avgs = {}
        for criterion in criteria_names:
            avg = sum(r['ratings'][criterion] for r in self.ratings) / len(self.ratings)
            criteria_avgs[criterion] = avg

        # Display as bar chart
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))

        criteria_labels = [
            'Legal\nAccuracy',
            'Completeness',
            'Citation\nQuality',
            'Clarity',
            'Practical\nUtility'
        ]

        bars = ax.bar(criteria_labels, list(criteria_avgs.values()),
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

        ax.set_ylim(0, 5)
        ax.set_ylabel('Average Rating (1-5)')
        ax.set_title('Average Ratings by Criterion')
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Neutral (3.0)')
        ax.legend()

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')

        st.pyplot(fig)

        # Export Options
        st.subheader("📥 تصدير البيانات / Export Data")

        col1, col2 = st.columns(2)

        with col1:
            # Export CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ تحميل CSV / Download CSV",
                csv,
                "expert_ratings.csv",
                "text/csv"
            )

        with col2:
            # Export JSON
            json_str = json.dumps(self.ratings, ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ تحميل JSON / Download JSON",
                json_str,
                "expert_ratings.json",
                "application/json"
            )

        # Statistical Analysis
        st.subheader("📈 التحليل الإحصائي / Statistical Analysis")

        st.markdown("""
        **Inter-Rater Reliability:**
        - Calculate Krippendorff's Alpha or Fleiss' Kappa
        - Measure agreement between experts
        - Include in thesis validation

        **Correlation Analysis:**
        - Correlation between different criteria
        - Relationship between experience and ratings
        - Include in results discussion
        """)

        if len(self.ratings) >= 10:
            st.success("""
            ✅ **عدد كافٍ من التقييمات للتحليل الإحصائي**

            يمكنك الآن حساب:
            - معامل الثبات بين المقيّمين (Inter-Rater Reliability)
            - الدلالة الإحصائية للنتائج
            - تضمين هذه البيانات في الرسالة
            """)
        else:
            remaining = 10 - len(self.ratings)
            st.info(f"⚠️ تحتاج {remaining} تقييمات إضافية للتحليل الإحصائي الكامل")


def main():
    """Test expert validation interface."""

    st.set_page_config(
        page_title="Expert Validation",
        page_icon="👨‍⚖️",
        layout="wide"
    )

    validator = ExpertValidationUI()

    # Sample query and answer for testing
    sample_query = "ما هي المسؤولية المدنية للموظف عن الأخطاء في العمل؟"
    sample_answer = """
    المسؤولية المدنية للموظف عن أخطائه في العمل تنظمها المواد 134-139 من قانون الموجبات والعقود اللبناني.

    الموظف يكون مسؤولاً عن الضرر الناتج عن خطئه في حالات:
    1. الخطأ الجسيم
    2. الخطأ المتعمد
    3. عدم الامتثال للتعليمات

    المسؤولية تقع على عاتق الموظف شخصياً في حالة الخطأ الشخصي.
    """
    sample_citations = ["المادة 134", "المادة 135", "المادة 139"]

    tab1, tab2 = st.tabs(["تقييم إجابة / Rate Answer", "لوحة التحكم / Dashboard"])

    with tab1:
        st.title("👨‍⚖️ تقييم إجابة النظام / Expert Validation")

        # Display query and answer
        st.markdown("### السؤال / Query")
        st.info(sample_query)

        st.markdown("### الإجابة / Answer")
        st.write(sample_answer)

        st.markdown("### الاستشهادات / Citations")
        st.write(", ".join(sample_citations))

        # Render validation interface
        validator.render_validation_interface(
            sample_query,
            sample_answer,
            sample_citations
        )

    with tab2:
        validator.render_validation_dashboard()


if __name__ == "__main__":
    main()
