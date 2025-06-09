"""
특허 설명 웹 스크래핑 모듈 테스트 스크립트

Google Patents 웹 사이트에서 특허 설명을 스크래핑하는 기능을 테스트합니다.
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.patent_search.description_scraper import (
        GooglePatentsDescriptionScraper,
        scrape_single_patent_description,
        scrape_multiple_patent_descriptions,
        ScrapingResult,
        ScrapingError,
    )

    print("✅ 웹 스크래핑 모듈 import 성공")
except ImportError as e:
    print(f"❌ 웹 스크래핑 모듈 import 실패: {e}")
    sys.exit(1)


def test_single_patent_scraping():
    """단일 특허 스크래핑 테스트"""
    print("\n🔍 단일 특허 스크래핑 테스트...")

    # 알려진 유효한 Google Patents URL
    test_url = "https://patents.google.com/patent/US11734097B1/en"

    try:
        print(f"테스트 URL: {test_url}")

        # 단일 특허 스크래핑
        result = scrape_single_patent_description(test_url)

        if result.success:
            print("✅ 스크래핑 성공!")
            print(f"📋 특허 ID: {result.patent_id}")
            print(f"🏷️ 제목: {result.title[:100]}...")
            print(f"📝 초록 길이: {len(result.abstract)} 문자")
            print(f"📄 설명 길이: {len(result.description)} 문자")
            print(f"⚖️ 청구항 수: {len(result.claims)}개")
            print(f"⏱️ 처리 시간: {result.processing_time:.2f}초")

            # 내용 샘플 출력
            if result.abstract:
                print(f"\n📝 초록 (처음 200자):\n{result.abstract[:200]}...")

            if result.description:
                print(f"\n📄 설명 (처음 300자):\n{result.description[:300]}...")

            if result.claims:
                print(f"\n⚖️ 첫 번째 청구항:\n{result.claims[0][:200]}...")

            return True
        else:
            print(f"❌ 스크래핑 실패: {result.error_message}")
            return False

    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        return False


def test_scraper_class():
    """스크래퍼 클래스 직접 테스트"""
    print("\n🏗️ 스크래퍼 클래스 직접 테스트...")

    try:
        # 스크래퍼 인스턴스 생성
        scraper = GooglePatentsDescriptionScraper(
            requests_per_second=1.0, timeout=15, max_retries=2  # 더 느린 속도로 테스트
        )

        test_url = "https://patents.google.com/patent/US11734097B1/en"
        print(f"테스트 URL: {test_url}")

        # 특허 ID 추출 테스트
        patent_id = scraper._extract_patent_id_from_url(test_url)
        print(f"📋 추출된 특허 ID: {patent_id}")

        # 스크래핑 실행
        result = scraper.scrape_patent_description(test_url)

        if result.success:
            print("✅ 클래스 스크래핑 성공!")

            # 통계 정보 확인
            stats = scraper.get_scraping_stats([result])
            print(f"📊 통계 정보:")
            for key, value in stats.items():
                print(f"   - {key}: {value}")

            return True
        else:
            print(f"❌ 클래스 스크래핑 실패: {result.error_message}")
            return False

    except Exception as e:
        print(f"❌ 클래스 테스트 중 오류: {e}")
        return False


def test_multiple_patents_scraping():
    """다중 특허 스크래핑 테스트"""
    print("\n🔄 다중 특허 스크래핑 테스트...")

    # 테스트용 특허 URL 목록 (실제 존재하는 특허들)
    test_urls = [
        "https://patents.google.com/patent/US11734097B1/en",
        "https://patents.google.com/patent/US10000000B1/en",  # 존재할 가능성
        "https://patents.google.com/patent/US9999999B1/en",  # 존재하지 않을 가능성
    ]

    try:
        print(f"테스트 URL 수: {len(test_urls)}개")

        # 진행률 콜백 함수
        def progress_callback(completed, total, result):
            status = "✅ 성공" if result.success else "❌ 실패"
            print(f"   진행률: {completed}/{total} - {result.patent_id}: {status}")
            if not result.success:
                print(f"      오류: {result.error_message}")

        # 다중 스크래핑 실행
        results = scrape_multiple_patent_descriptions(
            test_urls,
            max_workers=2,  # 적은 워커로 테스트
            progress_callback=progress_callback,
        )

        # 결과 분석
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\n📊 스크래핑 결과:")
        print(f"   - 총 특허: {len(results)}개")
        print(f"   - 성공: {len(successful)}개")
        print(f"   - 실패: {len(failed)}개")
        print(f"   - 성공률: {len(successful)/len(results)*100:.1f}%")

        # 성공한 결과 상세 정보
        for result in successful:
            print(f"\n✅ {result.patent_id}:")
            print(f"   - 제목: {result.title[:80]}...")
            print(f"   - 처리시간: {result.processing_time:.2f}초")
            print(f"   - 설명 길이: {len(result.description)} 문자")

        # 실패한 결과 정보
        for result in failed:
            print(f"\n❌ {result.patent_id}:")
            print(f"   - 오류: {result.error_message}")

        return len(successful) > 0

    except Exception as e:
        print(f"❌ 다중 스크래핑 테스트 중 오류: {e}")
        return False


def test_error_handling():
    """에러 처리 테스트"""
    print("\n🚫 에러 처리 테스트...")

    # 유효하지 않은 URL 테스트
    invalid_urls = [
        "https://patents.google.com/patent/INVALID123/en",
        "https://invalid-domain.com/patent/US123/",
        "not-a-url",
    ]

    success_count = 0

    for url in invalid_urls:
        try:
            print(f"\n테스트 URL: {url}")
            result = scrape_single_patent_description(url)

            if not result.success:
                print(f"✅ 예상대로 실패: {result.error_message[:100]}...")
                success_count += 1
            else:
                print(f"⚠️ 예상과 다르게 성공함")

        except Exception as e:
            print(f"✅ 예상대로 예외 발생: {str(e)[:100]}...")
            success_count += 1

    print(f"\n📊 에러 처리 테스트 결과: {success_count}/{len(invalid_urls)} 성공")
    return success_count == len(invalid_urls)


def test_rate_limiting():
    """속도 제한 테스트"""
    print("\n⏱️ 속도 제한 테스트...")

    try:
        # 빠른 속도 제한으로 스크래퍼 생성
        scraper = GooglePatentsDescriptionScraper(requests_per_second=2.0)

        test_url = "https://patents.google.com/patent/US11734097B1/en"

        # 연속 요청으로 속도 제한 확인
        times = []
        for i in range(3):
            start_time = time.time()

            # 실제 스크래핑 대신 rate_limiter만 테스트
            scraper.rate_limiter.wait_if_needed()

            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"   요청 {i+1}: {elapsed:.3f}초 대기")

        # 두 번째와 세 번째 요청은 속도 제한에 의해 지연되어야 함
        if times[1] > 0.4 and times[2] > 0.4:  # 2 req/sec = 0.5초 간격
            print("✅ 속도 제한이 올바르게 작동함")
            return True
        else:
            print("⚠️ 속도 제한이 예상대로 작동하지 않음")
            return False

    except Exception as e:
        print(f"❌ 속도 제한 테스트 중 오류: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("🚀 특허 설명 웹 스크래핑 모듈 테스트 시작")
    print("=" * 60)

    tests = [
        ("단일 특허 스크래핑", test_single_patent_scraping),
        ("스크래퍼 클래스 테스트", test_scraper_class),
        ("다중 특허 스크래핑", test_multiple_patents_scraping),
        ("에러 처리", test_error_handling),
        ("속도 제한", test_rate_limiting),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 실행 중 치명적 오류: {e}")
            results[test_name] = False

    print("\n" + "=" * 60)
    print("🎯 테스트 결과 요약:")

    for test_name, result in results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"  - {test_name}: {status}")

    passed = sum(results.values())
    total = len(results)

    print(f"\n📊 총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 모든 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패. 위의 오류 메시지를 확인하세요.")

    print(
        "\n💡 참고: 실제 웹 스크래핑은 네트워크 상태와 웹사이트 응답에 따라 결과가 달라질 수 있습니다."
    )


if __name__ == "__main__":
    main()
