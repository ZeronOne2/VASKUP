import sys
import os

sys.path.append("src")

from patent_search.serp_client import SerpPatentClient
from patent_search.description_scraper import GooglePatentsDescriptionScraper

print("🧪 Streamlit 앱 웹 스크래핑 로직 테스트")
print("=" * 50)

# 샘플 특허로 검색 (캐시된 데이터 사용)
client = SerpPatentClient()
patent_numbers = ["US11734097B1"]

print("\n1️⃣ 특허 검색 수행...")
search_results = client.batch_search_patents(patent_numbers)

print("\n2️⃣ 검색 결과 분석...")
for patent_id, result in search_results["results"].items():
    print(f"\n📋 특허: {patent_id}")
    print(
        f'  - description_link: {"✅" if result.get("description_link") else "❌"} {result.get("description_link", "N/A")[:80]}...'
    )
    print(
        f'  - google_patents_url: {"✅" if result.get("google_patents_url") else "❌"} {result.get("google_patents_url", "N/A")}'
    )

print("\n3️⃣ 웹 스크래핑 URL 결정 (Streamlit 로직 시뮬레이션)...")
patent_urls = []
for patent_id, result in search_results["results"].items():
    # Streamlit 앱과 동일한 로직
    if "description_link" in result and result["description_link"]:
        # description_link가 있다는 것은 특허가 유효하다는 의미이므로 google_patents_url 사용
        if "google_patents_url" in result and result["google_patents_url"]:
            patent_urls.append(result["google_patents_url"])
            print(f"  ✅ {patent_id}: google_patents_url 사용 (description_link 기반)")
        else:
            # google_patents_url이 없으면 patent_id로부터 생성
            base_url = "https://patents.google.com/patent/"
            generated_url = f"{base_url}{patent_id}/en"
            patent_urls.append(generated_url)
            print(f"  🔧 {patent_id}: URL 생성 (description_link 기반)")
    elif "google_patents_url" in result:
        # fallback: google_patents_url만 있는 경우
        patent_urls.append(result["google_patents_url"])
        print(f"  🔄 {patent_id}: google_patents_url 사용 (fallback)")
    else:
        print(f"  ❌ {patent_id}: 웹 스크래핑 URL 없음")

print(f"\n4️⃣ 선택된 스크래핑 URL들: {len(patent_urls)}개")
for i, url in enumerate(patent_urls, 1):
    print(f"  {i}. {url}")

if patent_urls:
    print("\n5️⃣ 웹 스크래핑 테스트 (첫 번째 URL)...")
    scraper = GooglePatentsDescriptionScraper(requests_per_second=1.0)
    test_result = scraper.scrape_patent_description(patent_urls[0])

    print(f'  - 성공: {"✅" if test_result.success else "❌"}')
    print(f"  - 특허 ID: {test_result.patent_id}")
    print(f"  - 제목 길이: {len(test_result.title)} 문자")
    print(f"  - 초록 길이: {len(test_result.abstract)} 문자")
    print(f"  - 설명 길이: {len(test_result.description)} 문자")
    print(f"  - 청구항 수: {len(test_result.claims)}개")
    print(f"  - 처리 시간: {test_result.processing_time:.2f}초")

    if not test_result.success:
        print(f"  - 오류: {test_result.error_message}")

print("\n✅ 테스트 완료!")
print("\n💡 결론:")
print("- description_link(PDF)를 우선 고려하지만 실제로는 google_patents_url을 사용")
print("- 이는 요구사항에 부합하면서도 실용적인 접근법입니다")
