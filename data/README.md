# Data — 합성 예제 전용

어떤 원본 CSV/XLS/XLSX 파일도 이 저장소에 포함하지 않는다.

`python -m src.synthetic`는 실제 지역이나 거래와 무관한 합성 자료를 생성해 개수와 컬럼명만 표시한다. `python -m src.evaluate`는 같은 데이터를 메모리에서 생성한다.

| 컬럼 | 의미 |
| --- | --- |
| zone | 가상 구역 ID (Z00~Z39) |
| month | 예제 상대 월 (1~24) |
| price_pressure | 임의의 가격압력 (0~1) |
| collateral_pressure | 임의의 담보압력 (0~1) |
| insurance_access | 임의의 보험 접근성 (0~1; 높을수록 접근 용이) |
| housing_proxy | 임의의 주거 구조 참고지표 (0~1) |
| future_rate | 가상의 다음 달 목표 비율 (0~1) |

가중치 및 생성식은 교육용 임의 값이다. 외부 원자료에서 통계적 분포나 실제 지명·행 데이터를 추출하지 않았다. 실제 위험률/확률/지역 안전성과 무관하다.

`src/contract_demo.py`의 별도 합성 계약 예제는 `contract`, `zone`, `price_ratio`, `collateral`, `fictional_label`로 구성한다. 지역과 계약 ID는 완전 가상이며 A/B 분할 개념만 시연한다. 원래의 실거래 기반 파일과 동일한 스키마가 아니다. `src/score_review.py`는 정규화된 여섯 가상 지표로 발표 화면의 가중합을 계산한다. 원 데이터 접근과 무관하다. 원본 파일의 출처·구성은 [데이터 카탈로그](../docs/data-catalog.md)를 참고한다.

