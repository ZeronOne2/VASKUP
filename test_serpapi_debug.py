"""
SerpAPI 연결 디버깅 및 테스트 스크립트
"""

import os
import sys
import requests
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.patent_search.serp_client import SerpPatentClient, PatentSearchError

    print("✅ SerpAPI 모듈 import 성공")
except ImportError as e:
    print(f"❌ SerpAPI 모듈 import 실패: {e}")
    sys.exit(1)


def test_api_key():
    """API 키 유효성 테스트"""
    print("\n🔑 API 키 테스트...")

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print("❌ SERPAPI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return False

    if api_key.startswith("your_"):
        print("❌ API 키가 기본값(your_serpapi_key_here)으로 설정되어 있습니다.")
        return False

    print(f"✅ API 키 발견: ***{api_key[-6:]}")
    return True


def test_basic_serpapi_connection():
    """기본 SerpAPI 연결 테스트"""
    print("\n🌐 SerpAPI 기본 연결 테스트...")

    api_key = os.getenv("SERPAPI_API_KEY")
    base_url = "https://serpapi.com/search"

    # 간단한 Google 검색으로 API 연결 확인
    params = {"engine": "google", "q": "test", "api_key": api_key, "num": 1}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        print(f"📊 응답 상태: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print(f"❌ API 오류: {data['error']}")
                return False
            else:
                print("✅ SerpAPI 기본 연결 성공")
                return True
        else:
            print(f"❌ HTTP 오류: {response.status_code}")
            print(f"응답 내용: {response.text[:200]}...")
            return False

    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return False


def test_google_patents_engine():
    """Google Patents 엔진 테스트"""
    print("\n🔬 Google Patents 엔진 테스트...")

    api_key = os.getenv("SERPAPI_API_KEY")
    base_url = "https://serpapi.com/search"

    # 알려진 유효한 특허로 테스트
    test_patents = [
        "patent/US11734097B1/en",  # 실제 존재하는 특허
        "patent/US10000000B1/en",  # 존재할 가능성이 높은 특허
    ]

    for patent_id in test_patents:
        print(f"\n🧪 테스트 특허: {patent_id}")

        params = {
            "engine": "google_patents_details",
            "patent_id": patent_id,
            "api_key": api_key,
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            print(f"📊 응답 상태: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    print(f"❌ API 오류: {data['error']}")
                else:
                    print(f"✅ 성공: {data.get('title', '제목 없음')[:50]}...")
                    return True
            else:
                print(f"❌ HTTP 오류: {response.status_code}")
                print(f"응답 내용: {response.text[:300]}...")

        except Exception as e:
            print(f"❌ 요청 실패: {e}")

    return False


def test_patent_search_formats():
    """다양한 특허 번호 형식 테스트"""
    print("\n📋 특허 번호 형식 테스트...")

    try:
        client = SerpPatentClient()

        # 다양한 형식의 특허 번호 변환 테스트
        test_formats = [
            "US11734097B1",
            "US10000000B1",
            "US20200000000A1",
            "EP1234567A1",
            "patent/US11734097B1/en",  # 이미 변환된 형식
        ]

        for patent_num in test_formats:
            converted = client.convert_patent_number_to_id(patent_num)
            print(f"📝 {patent_num} → {converted}")

        return True

    except Exception as e:
        print(f"❌ 변환 테스트 실패: {e}")
        return False


def test_actual_patent_search():
    """실제 특허 검색 테스트"""
    print("\n🔍 실제 특허 검색 테스트...")

    try:
        client = SerpPatentClient()

        # 확실히 존재하는 잘 알려진 특허들
        known_patents = [
            "US11734097B1",  # 실제 존재하는 특허
        ]

        for patent in known_patents:
            print(f"\n🧪 검색 중: {patent}")
            try:
                result = client.search_patent(patent)
                print(f"✅ 성공: {result['title'][:50]}...")
                print(f"📅 출원일: {result['application_date']}")
                print(f"👨‍💼 발명자: {', '.join(result['inventor'][:2])}")
                return True

            except PatentSearchError as e:
                print(f"❌ 검색 실패: {e}")
                continue

        return False

    except Exception as e:
        print(f"❌ 클라이언트 초기화 실패: {e}")
        return False


def main():
    """메인 디버깅 실행"""
    print("🚀 SerpAPI 디버깅 및 테스트 시작")
    print("=" * 60)

    tests = [
        ("API 키 확인", test_api_key),
        ("기본 SerpAPI 연결", test_basic_serpapi_connection),
        ("Google Patents 엔진", test_google_patents_engine),
        ("특허 번호 형식", test_patent_search_formats),
        ("실제 특허 검색", test_actual_patent_search),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 실행 중 오류: {e}")
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


if __name__ == "__main__":
    main()
