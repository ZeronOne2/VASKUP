import sys
import os

sys.path.append("src")

from patent_search.serp_client import SerpPatentClient
from patent_search.description_scraper import GooglePatentsDescriptionScraper

print("🔍 스크래핑 결과 통합 디버깅")
print("=" * 50)

# 캐시된 특허로 테스트
patent_numbers = ["US20210390793A1", "US11630280B2", "US11630282B2"]
client = SerpPatentClient()

print("1️⃣ 특허 검색 수행...")
search_results = client.batch_search_patents(patent_numbers)

print("\n2️⃣ 검색 결과 확인...")
for patent_id, result in search_results["results"].items():
    print(f"\n📋 특허: {patent_id}")
    print(f'  - 제목: {result.get("title", "N/A")[:50]}...')
    print(f'  - description_link: {result.get("description_link", "N/A")[:80]}...')

# Description links 수집
patent_urls = []
url_to_patent_mapping = {}  # URL과 특허 ID 매핑

for patent_id, result in search_results["results"].items():
    if "description_link" in result and result["description_link"]:
        url = result["description_link"]
        patent_urls.append(url)
        url_to_patent_mapping[url] = patent_id
        print(f"  ✅ URL 추가: {patent_id} -> {url[:80]}...")

print(f"\n3️⃣ 웹 스크래핑 수행... ({len(patent_urls)}개 URL)")
scraper = GooglePatentsDescriptionScraper(requests_per_second=1.0)

if patent_urls:
    scraping_results = scraper.batch_scrape_patents(patent_urls, max_workers=2)

    print("\n4️⃣ 스크래핑 결과 분석...")
    for i, scraping_result in enumerate(scraping_results):
        print(f"\n🕷️ 스크래핑 결과 {i+1}:")
        print(f"  - URL: {scraping_result.url[:80]}...")
        print(f'  - patent_id: "{scraping_result.patent_id}"')
        print(f"  - 성공: {scraping_result.success}")
        print(
            f"  - 제목: {scraping_result.title[:50]}..."
            if scraping_result.title
            else "  - 제목: 없음"
        )
        print(f"  - 설명 길이: {len(scraping_result.description)} 문자")
        print(f"  - 청구항 개수: {len(scraping_result.claims)} 개")

        # URL로 실제 특허 ID 찾기
        actual_patent_id = url_to_patent_mapping.get(scraping_result.url, "알 수 없음")
        print(f"  - 실제 특허 ID: {actual_patent_id}")

    print("\n5️⃣ 매칭 로직 테스트...")
    scraping_success = 0

    for scraping_result in scraping_results:
        if scraping_result.success:
            print(f"\n🔄 매칭 시도: {scraping_result.patent_id}")

            # 기존 매칭 로직
            matched_old = False
            for patent_id, search_result in search_results["results"].items():
                if (
                    scraping_result.patent_id in patent_id
                    or patent_id in scraping_result.patent_id
                ):
                    print(f"  ✅ 기존 로직으로 매칭: {patent_id}")
                    matched_old = True
                    break

            # 개선된 매칭 로직 (URL 기반)
            matched_new = False
            actual_patent_id = url_to_patent_mapping.get(scraping_result.url)
            if actual_patent_id and actual_patent_id in search_results["results"]:
                print(f"  ✅ URL 기반 매칭: {actual_patent_id}")
                matched_new = True
                scraping_success += 1

            if not matched_old and not matched_new:
                print(f"  ❌ 매칭 실패")
                print(f'    - 스크래핑 patent_id: "{scraping_result.patent_id}"')
                print(
                    f'    - 검색 결과 patent_ids: {list(search_results["results"].keys())}'
                )
                print(f"    - URL: {scraping_result.url[:80]}...")

    print(f"\n📊 매칭 결과: {scraping_success}/{len(scraping_results)} 성공")

    if scraping_success == 0:
        print("\n💡 권장사항: URL 기반 매칭 로직을 사용해야 합니다!")
    else:
        print("\n✅ 매칭이 성공적으로 작동합니다!")

else:
    print("❌ 스크래핑할 URL이 없습니다.")
