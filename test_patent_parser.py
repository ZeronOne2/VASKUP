import sys
import os

sys.path.append("src")

import json
from patent_search.patent_parser import (
    PatentDataProcessor,
    Patent,
    DocumentChunk,
    format_patent_id,
    parse_description,
)
from patent_search.serp_client import SerpPatentClient
from patent_search.description_scraper import GooglePatentsDescriptionScraper

print("🔍 특허 데이터 처리 및 청킹 모듈 테스트")
print("=" * 60)


def test_format_patent_id():
    """특허 ID 변환 함수 테스트"""
    print("\n1️⃣ 특허 ID 변환 테스트")

    test_cases = [
        "US11734097B1",
        "US20210390793A1",
        "patent/US11630280B2/en",  # 이미 변환된 형식
        "US 11630282 B2",  # 공백 포함
        "",  # 빈 문자열
    ]

    for patent_num in test_cases:
        formatted = format_patent_id(patent_num)
        print(f"  '{patent_num}' -> '{formatted}'")


def test_parse_description():
    """HTML 설명 파싱 테스트"""
    print("\n2️⃣ HTML 설명 파싱 테스트")

    # 샘플 HTML (실제와 유사한 구조)
    html_content = """
    <div>
        <span class="google-src-text">원본 텍스트 (제거 대상)</span>
        <p>This is the English translation of the patent description.</p>
        <span class="google-src-text">더 많은 원본 텍스트</span>
        <p>Multiple paragraphs with important technical details.</p>
    </div>
    """

    parsed = parse_description(html_content)
    print(f"  파싱 결과: '{parsed}'")
    print(f"  길이: {len(parsed)} 문자")


def test_patent_class():
    """Patent 클래스 테스트"""
    print("\n3️⃣ Patent 클래스 테스트")

    # 기본 Patent 객체 생성
    patent = Patent(
        patent_number="US11734097B1",
        patent_id="",  # __post_init__에서 자동 생성됨
        title="Test Patent Title",
        abstract="This is a test patent abstract with technical details.",
        claims=["First claim of the patent", "Second claim of the patent"],
        description="Original description from SERP API",
        filing_date="2021-01-27",
        publication_date="2023-08-22",
    )

    print(f"  특허번호: {patent.patent_number}")
    print(f"  자동 생성된 patent_id: {patent.patent_id}")
    print(f"  완전성 검사: {patent.is_complete}")
    print(f"  효과적인 설명 길이: {len(patent.effective_description)} 문자")
    print(f"  효과적인 청구항 수: {len(patent.effective_claims)}개")

    # 웹 스크래핑 결과 시뮬레이션
    patent.scraped_description = (
        "Enhanced description from web scraping with more details"
    )
    patent.scraping_success = True

    print(f"  스크래핑 후 효과적인 설명: {len(patent.effective_description)} 문자")
    print(f"  스크래핑 성공: {patent.scraping_success}")


def test_patent_chunker():
    """특허 청킹 테스트"""
    print("\n4️⃣ 특허 청킹 테스트")

    # 테스트용 긴 설명 생성
    long_description = " ".join(
        [
            f"This is paragraph {i} of a very long patent description."
            for i in range(1, 100)  # 긴 텍스트 생성
        ]
    )

    patent = Patent(
        patent_number="US11734097B1",
        patent_id="patent/US11734097B1/en",
        title="Advanced Machine Learning System for Patent Analysis",
        abstract="A comprehensive system for analyzing patents using advanced ML techniques.",
        claims=[
            "A method for processing patent data comprising steps of data collection and analysis",
            "The method of claim 1, wherein the analysis includes natural language processing",
            "A system implementing the method of claim 1 with distributed computing resources",
        ],
        description=long_description,
        filing_date="2021-01-27",
        publication_date="2023-08-22",
    )

    # 청킹 수행
    from patent_search.patent_parser import PatentChunker

    chunker = PatentChunker(chunk_size=500, chunk_overlap=100)  # 테스트용 작은 크기
    chunks = chunker.chunk_patent(patent)

    print(f"  생성된 청크 수: {len(chunks)}개")

    # 섹션별 청크 분포
    distribution = {}
    for chunk in chunks:
        section = chunk.section
        distribution[section] = distribution.get(section, 0) + 1

    print("  섹션별 청크 분포:")
    for section, count in distribution.items():
        print(f"    {section}: {count}개")

    # 각 청크 미리보기
    print("\n  청크 미리보기:")
    for i, chunk in enumerate(chunks[:5]):  # 처음 5개만
        preview = (
            chunk.content[:50] + "..." if len(chunk.content) > 50 else chunk.content
        )
        print(f"    청크 {i+1} ({chunk.section}): {preview}")
        print(f"      청크 ID: {chunk.chunk_id}")


def test_real_data_processing():
    """실제 데이터로 종합 테스트"""
    print("\n5️⃣ 실제 데이터 종합 처리 테스트")

    # 캐시된 데이터 로드
    try:
        with open("src/cache/patent_cache.json", "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        print(f"  캐시된 특허 수: {len(cache_data)}개")

        # 처음 3개 특허로 테스트 (실제 캐시 구조에 맞게 변환)
        test_patents = {}
        count = 0
        for cache_key, cache_item in cache_data.items():
            if count >= 3:
                break

            # 캐시 구조: cache_item['data']에 실제 특허 데이터가 있음
            patent_data = cache_item.get("data", {})

            # 특허번호 추출 (publication_number나 google_patents_url에서)
            patent_number = None
            if (
                "raw_data" in patent_data
                and "publication_number" in patent_data["raw_data"]
            ):
                patent_number = patent_data["raw_data"]["publication_number"]
            elif "google_patents_url" in patent_data:
                # URL에서 특허번호 추출: https://patents.google.com/patent/US11734097B1/en
                import re

                match = re.search(
                    r"/patent/([^/]+)/", patent_data["google_patents_url"]
                )
                if match:
                    patent_number = match.group(1)

            if patent_number:
                # 캐시 구조를 SerpAPI 결과 구조로 변환
                converted_data = {
                    "title": patent_data.get("title", ""),
                    "abstract": patent_data.get("abstract", ""),
                    "claims": patent_data.get("claims", []),
                    "description": "",  # 캐시에는 description이 직접 없음
                    "description_link": patent_data.get("description_link", ""),
                    "filing_date": patent_data.get(
                        "application_date", ""
                    ),  # filing_date -> application_date 매핑
                    "publication_date": patent_data.get("publication_date", ""),
                    "application_date": patent_data.get("application_date", ""),
                    "status": patent_data.get("status", ""),
                    "inventor": patent_data.get("inventor", []),
                    "assignee": patent_data.get("assignee", []),
                    "classifications": patent_data.get("classifications", []),
                    "citations": patent_data.get("citations", {}),
                    "google_patents_url": patent_data.get("google_patents_url", ""),
                    "search_timestamp": patent_data.get("search_timestamp", ""),
                    "raw_data": patent_data.get("raw_data", {}),
                }

                test_patents[patent_number] = converted_data
                count += 1

        if not test_patents:
            print("  변환 가능한 특허 데이터가 없습니다.")
            return None

        search_results = {"results": test_patents}

        print(f"  변환된 특허 수: {len(test_patents)}개")
        for patent_num in test_patents.keys():
            print(f"    - {patent_num}")

        # 데이터 처리기 생성
        processor = PatentDataProcessor(chunk_size=800, chunk_overlap=150)

        # 처리 수행
        patents, chunks = processor.process_search_results(search_results)

        print(f"  처리된 특허 수: {len(patents)}개")
        print(f"  생성된 청크 수: {len(chunks)}개")

        if len(chunks) == 0:
            print("  ⚠️ 청크가 생성되지 않았습니다. 데이터 내용을 확인합니다:")
            for patent in patents:
                print(f"    특허 {patent.patent_number}:")
                print(f"      제목 길이: {len(patent.title)} 문자")
                print(f"      초록 길이: {len(patent.abstract)} 문자")
                print(f"      청구항 수: {len(patent.claims)}개")
                print(f"      설명 길이: {len(patent.description)} 문자")
                print(
                    f"      스크래핑된 설명 길이: {len(patent.scraped_description)} 문자"
                )
                print(f"      완전성: {patent.is_complete}")

        # 구조화된 데이터 생성
        structured_data = processor.combine_to_structured_format(patents, chunks)

        print(f"  메타데이터:")
        print(f"    전체 특허: {structured_data['metadata']['total_patents']}개")
        print(f"    전체 청크: {structured_data['metadata']['total_chunks']}개")
        print(f"    처리 시간: {structured_data['metadata']['processing_timestamp']}")
        print(f"    청크 분포: {structured_data['metadata']['chunk_distribution']}")

        # 샘플 특허 상세 정보
        if patents:
            sample_patent = patents[0]
            print(f"\n  샘플 특허 ({sample_patent.patent_number}):")
            print(f"    제목: {sample_patent.title[:50]}...")
            print(f"    초록: {sample_patent.abstract[:50]}...")
            print(f"    청구항 수: {len(sample_patent.effective_claims)}개")
            print(f"    설명 길이: {len(sample_patent.effective_description):,} 문자")
            print(f"    설명 청크 수: {len(sample_patent.description_chunks)}개")
            print(f"    완전성: {sample_patent.is_complete}")

        return structured_data

    except FileNotFoundError:
        print("  캐시 파일이 없습니다. 실제 데이터 테스트를 건너뜁니다.")
        return None
    except Exception as e:
        print(f"  실제 데이터 테스트 중 오류: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """메인 테스트 실행"""
    try:
        test_format_patent_id()
        test_parse_description()
        test_patent_class()
        test_patent_chunker()
        structured_data = test_real_data_processing()

        print("\n✅ 모든 테스트 완료!")

        # 결과를 파일로 저장 (선택적)
        if structured_data:
            output_file = "test_patent_processing_result.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            print(f"📁 테스트 결과가 {output_file}에 저장되었습니다.")

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
