#!/usr/bin/env python3
"""설명 스크래핑 및 청킹 테스트"""

import json
import sys
import os

sys.path.append("src")

from patent_search.patent_parser import PatentDataProcessor
from patent_search.description_scraper import GooglePatentsDescriptionScraper


def test_description_scraping_and_chunking():
    """실제 설명 스크래핑 후 청킹 테스트"""

    print("🔍 설명 스크래핑 및 청킹 테스트 시작")
    print("=" * 60)

    # 1. 캐시에서 description_link가 있는 샘플 찾기
    with open("src/cache/patent_cache.json", "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    samples_with_links = []
    for key, item in cache_data.items():
        if "data" not in item:
            continue
        data = item["data"]
        if "description_link" in data and data["description_link"]:
            patent_number = None
            if "raw_data" in data and "publication_number" in data["raw_data"]:
                patent_number = data["raw_data"]["publication_number"]
            elif "google_patents_url" in data:
                import re

                match = re.search(r"/patent/([^/]+)/", data["google_patents_url"])
                if match:
                    patent_number = match.group(1)

            if patent_number:
                samples_with_links.append(
                    (key, patent_number, data["description_link"])
                )

    print(f"Description_link가 있는 샘플: {len(samples_with_links)}개")

    if not samples_with_links:
        print("❌ Description_link가 있는 샘플을 찾을 수 없습니다.")
        return

    # 첫 번째 샘플로 테스트
    sample_key, patent_number, description_link = samples_with_links[0]
    print(f"테스트 샘플: {patent_number}")
    print(f"Description link: {description_link}")

    # 2. 실제 웹 스크래핑 수행
    print(f"\n📡 웹 스크래핑 중...")
    scraper = GooglePatentsDescriptionScraper()

    try:
        # 단일 URL 스크래핑
        scraping_result = scraper.scrape_patent_description(description_link)

        if not scraping_result.success:
            print(f"❌ 스크래핑 실패: {scraping_result.error_message}")
            return

        description = scraping_result.description

        print(f"✅ 스크래핑 성공: {len(description):,} 문자")
        print(f"미리보기: {description[:200]}...")

    except Exception as e:
        print(f"❌ 스크래핑 오류: {e}")
        return

    # 3. 캐시 데이터를 SerpAPI 형식으로 변환
    cache_item = cache_data[sample_key]["data"]

    converted_data = {
        "title": cache_item.get("title", ""),
        "abstract": cache_item.get("abstract", ""),
        "claims": cache_item.get("claims", []),
        "description": description,  # 스크래핑된 설명 사용
        "description_link": cache_item.get("description_link", ""),
        "filing_date": cache_item.get("application_date", ""),
        "publication_date": cache_item.get("publication_date", ""),
        "application_date": cache_item.get("application_date", ""),
        "status": cache_item.get("status", ""),
        "inventor": cache_item.get("inventor", []),
        "assignee": cache_item.get("assignee", []),
        "classifications": cache_item.get("classifications", []),
        "citations": cache_item.get("citations", {}),
        "google_patents_url": cache_item.get("google_patents_url", ""),
        "search_timestamp": cache_item.get("search_timestamp", ""),
        "raw_data": cache_item.get("raw_data", {}),
    }

    search_results = {"results": {patent_number: converted_data}}

    # 4. 데이터 처리 및 청킹
    print(f"\n🔪 텍스트 청킹 처리...")
    processor = PatentDataProcessor(chunk_size=1200, chunk_overlap=200)

    try:
        patents, chunks = processor.process_search_results(search_results)

        print(f"✅ 처리 완료:")
        print(f"  특허 수: {len(patents)}")
        print(f"  청크 수: {len(chunks)}")

        if patents:
            patent = patents[0]
            print(f"\n📊 특허 정보:")
            print(f"  번호: {patent.patent_number}")
            print(f"  제목: {patent.title}")
            print(f"  초록 길이: {len(patent.abstract):,} 문자")
            print(f"  청구항 수: {len(patent.claims)}")
            print(f"  설명 길이: {len(patent.description):,} 문자")
            print(f"  완전성: {patent.is_complete}")

            print(f"\n📦 청크 분석:")
            chunk_types = {}
            for chunk in chunks:
                chunk_type = chunk.section  # section 필드 직접 사용
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            for chunk_type, count in sorted(chunk_types.items()):
                print(f"  {chunk_type}: {count}개")

            # 설명 청크들 미리보기
            description_chunks = [
                c
                for c in chunks
                if c.section == "description"  # section 필드 직접 사용
            ]
            if description_chunks:
                print(f"\n📝 설명 청크 미리보기 (총 {len(description_chunks)}개):")
                for i, chunk in enumerate(description_chunks[:3]):  # 처음 3개만
                    print(f"  청크 {i+1}: {chunk.content[:100]}...")
                    print(f"    길이: {len(chunk.content)} 문자")
                    print(f"    청크 ID: {chunk.chunk_id}")

            # 결과 저장
            structured_data = processor.combine_to_structured_format(patents, chunks)

            with open(
                "description_chunking_test_result.json", "w", encoding="utf-8"
            ) as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)

            print(
                f"\n💾 결과가 description_chunking_test_result.json에 저장되었습니다."
            )

            return True

    except Exception as e:
        print(f"❌ 처리 오류: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_description_scraping_and_chunking()
