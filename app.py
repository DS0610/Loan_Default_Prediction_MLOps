import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

# FastAPI 서버 주소
FASTAPI_URL = "http://127.0.0.1:8001/predict"
LOG_FILE = "prediction_logs.csv"

# Streamlit 페이지 설정
st.set_page_config(layout="wide", page_title="대출 승인 예측 시스템")

# 사이드바
page = st.sidebar.radio("페이지 선택", ["대출 승인 예측", "서비스 모니터링"])

# 대출 승인 예측 페이지
if page == "대출 승인 예측":
    st.title("대출 승인 예측 서비스 🤖")
    st.write("새로운 대출 신청자의 정보를 입력하고 승인 여부를 예측합니다.")

    with st.form("prediction_form"):
        st.header("신청자 정보 입력")

        # train.py에서 사용된 범주형 변수들의 실제 값 목록 (예시)
        grade_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        sub_grade_options = [f'{g}{i}' for g in grade_options for i in range(1, 6)]
        emp_length_options = {
            '< 1 year': '1년 미만', '1 year': '1년', '2 years': '2년', '3 years': '3년',
            '4 years': '4년', '5 years': '5년', '6 years': '6년', '7 years': '7년',
            '8 years': '8년', '9 years': '9년', '10+ years': '10년 이상'
        }
        home_ownership_options = {
            'RENT': '전/월세', 'OWN': '자가', 'MORTGAGE': '주택 담보대출',
            'ANY': '기타', 'OTHER': '기타', 'NONE': '없음'
        }
        verification_status_options = {
            'Verified': '소득 증빙 완료', 'Source Verified': '증빙 서류 제출', 'Not Verified': '증빙 미완료'
        }
        purpose_options = {
            'debt_consolidation': '부채 통합', 'credit_card': '신용카드 대금',
            'home_improvement': '주택 개선', 'other': '기타', 'major_purchase': '고가 상품 구매',
            'small_business': '사업 자금', 'car': '자동차 구매', 'medical': '의료비',
            'moving': '이사 비용', 'vacation': '여행 자금', 'house': '주택 구매',
            'wedding': '결혼 자금', 'renewable_energy': '신재생에너지'
        }

        # UI를 3개의 컬럼으로 구성
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("대출 기본 정보")
            loan_amnt_krw = st.number_input("대출 신청 금액 (만원)", min_value=100, max_value=10000, value=1000, step=50, help="신청할 대출 금액을 만원 단위로 입력하세요. (예: 1000만원)")
            # FastAPI 전송을 위해 달러로 환산
            loan_amnt = loan_amnt_krw * 10000 / 1300 # 환율 1300원 가정
            
            term = st.selectbox("대출 기간 (개월)", [36, 60], help="대출 상환 기간을 선택하세요.")
            int_rate = st.slider("이자율 (%)", 4.0, 20.0, 10.0, 0.1, help="예상되는 대출 이자율입니다.")
            grade = st.selectbox("대출 등급 (Grade)", grade_options, help="신청자의 신용 등급입니다.")
            sub_grade = st.selectbox("세부 등급 (Sub-Grade)", sub_grade_options, help="신용 등급의 세부 단계입니다.")
            
        with col2:
            st.subheader("신청자 재정 정보")
            annual_inc_krw = st.number_input("연 소득 (만원)", min_value=1500, value=5000, step=100, help="신청자의 세전 연간 소득을 만원 단위로 입력하세요.")
            # FastAPI 전송을 위해 달러로 환산
            annual_inc = annual_inc_krw * 10000 / 1300

            emp_length_display = st.selectbox("재직 기간", list(emp_length_options.values()), help="신청자의 현재 직장 재직 기간입니다.")
            # FastAPI 전송을 위해 원본 값으로 변환
            emp_length = [k for k, v in emp_length_options.items() if v == emp_length_display][0]

            home_ownership_display = st.selectbox("주택 소유 형태", list(home_ownership_options.values()), help="신청자의 주택 소유 상태입니다.")
            home_ownership = [k for k, v in home_ownership_options.items() if v == home_ownership_display][0]

            verification_status_display = st.selectbox("소득 증빙 상태", list(verification_status_options.values()), help="소득 증빙 여부 및 방식입니다.")
            verification_status = [k for k, v in verification_status_options.items() if v == verification_status_display][0]

            purpose_display = st.selectbox("대출 목적", list(purpose_options.values()), help="대출금의 주된 사용 목적입니다.")
            purpose = [k for k, v in purpose_options.items() if v == purpose_display][0]

        with col3:
            st.subheader("신용 및 부채 정보")
            dti = st.slider("총부채원리금상환비율 (DTI, %)", 0.0, 70.0, 35.0, 0.1, help="소득 대비 총 부채 원리금의 비율입니다.")
            
            revol_bal_krw = st.number_input("리볼빙 잔액 (만원)", min_value=0, value=500, step=10, help="신용카드의 리볼빙 서비스 잔액을 만원 단위로 입력하세요.")
            revol_bal = revol_bal_krw * 10000 / 1300

            revol_util = st.slider("리볼빙 한도 사용률 (%)", 0.0, 100.0, 50.0, 0.1, help="리볼빙 한도 대비 사용 금액의 비율입니다.")
            inq_last_6mths = st.number_input("최근 6개월간 신용 조회 수", min_value=0, value=1, help="최근 6개월간 신용 정보 조회 횟수입니다.")
            delinq_2yrs = st.number_input("2년간 연체 기록", min_value=0, value=0, help="지난 2년간 30일 이상 연체한 횟수입니다.")
            pub_rec = st.number_input("공개된 부정 기록 수", min_value=0, value=0, help="파산, 압류 등 공개된 부정적 신용 기록의 수입니다.")
            open_acc = st.number_input("현재 보유중인 대출/카드 수", min_value=0, value=10, help="현재 개설된 대출, 신용카드 등의 수입니다.")
            total_acc = st.number_input("총 개설했던 금융 계좌 수", min_value=1, value=25, help="지금까지 개설했던 모든 대출, 신용카드의 수입니다.")
            
        # 제출 버튼
        submit_button = st.form_submit_button(label="예측 실행")

    if submit_button:
        # FastAPI에 보낼 데이터 구성
        request_data = {
            "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "grade": grade,
            "sub_grade": sub_grade, "emp_length": emp_length, "home_ownership": home_ownership,
            "annual_inc": annual_inc, "verification_status": verification_status,
            "purpose": purpose, "dti": dti, "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_last_6mths, "open_acc": open_acc, "pub_rec": pub_rec,
            "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc
        }

        with st.spinner('모델이 예측 중입니다...'):
            try:
                # FastAPI 서버에 POST 요청
                response = requests.post(FASTAPI_URL, json=request_data)
                response.raise_for_status()  # 요청 실패 시 예외 발생
                result = response.json()

                st.subheader("예측 결과")
                if result['prediction_code'] == 0:
                    st.success(f"**결과: {result['prediction']}**")
                else:
                    st.error(f"**결과: {result['prediction']}**")

                st.info(f"**부실(거절) 확률: {result['probability_of_default']}**")
            except requests.exceptions.RequestException as e:
                st.error(f"API 서버 연결 오류: {e}")
                st.warning("FastAPI 서버(`main.py`)가 실행 중인지 확인해주세요.")
            except Exception as e:
                st.error(f"예측 중 알 수 없는 오류 발생: {e}")

# 서비스 모니터링 페이지
elif page == "서비스 모니터링":
    st.title("서비스 모니터링 대시보드 📊")

    if not os.path.exists(LOG_FILE):
        st.warning("아직 예측 로그가 없습니다. '대출 승인 예측' 페이지에서 예측을 먼저 실행해주세요.")
    else:
        log_df = pd.read_csv(LOG_FILE)
        st.subheader("최근 예측 로그")
        st.dataframe(log_df.tail(10))

        st.subheader("모니터링 차트")
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])

        col1, col2 = st.columns(2)
        with col1:
            # 예측 결과 분포 (파이 차트)
            prediction_counts = log_df['prediction'].value_counts().reset_index()
            prediction_counts.columns = ['prediction', 'count']
            prediction_counts['prediction'] = prediction_counts['prediction'].map({0: '승인', 1: '거절'})
            fig_pie = px.pie(prediction_counts, names='prediction', values='count',
                             title='예측 결과 분포 (승인 vs 거절)',
                             color='prediction',
                             color_discrete_map={'승인':'#636EFA', '거절':'#EF553B'})
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            # 시간별 예측 수 (라인 차트)
            log_df['date'] = log_df['timestamp'].dt.date
            requests_per_day = log_df.groupby('date').size().reset_index(name='count')
            fig_line = px.line(requests_per_day, x='date', y='count',
                               title='일별 예측 요청 수', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("주요 입력 변수 분포")
        feature_options = [col for col in log_df.columns if col not in ['prediction', 'probability', 'timestamp', 'date']]
        selected_feature = st.selectbox("분포를 확인할 변수 선택", feature_options)
        if selected_feature:
            fig_hist = px.histogram(log_df, x=selected_feature,
                                    title=f"'{selected_feature}'의 분포",
                                    nbins=30, color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig_hist, use_container_width=True)
