import sys
import os

sys.path.append("src")

import json
from patent_search.serp_client import SerpPatentClient

print("🔍 테스트용 특허 캐시 데이터 생성")
print("=" * 40)

# 테스트용 특허 번호들
test_patents = ["US20210390793A1", "US11630280B2", "US11630282B2"]

print(f"특허 검색 수행: {test_patents}")

# SerpAPI 클라이언트 생성
client = SerpPatentClient()

# 배치 검색 수행
search_results = client.batch_search_patents(test_patents)

if search_results["success"]:
    print(f'검색 성공: {len(search_results["results"])}개 특허')

    # 캐시 파일로 저장
    with open("patent_cache.json", "w", encoding="utf-8") as f:
        json.dump(search_results["results"], f, ensure_ascii=False, indent=2)

    print("✅ patent_cache.json 파일이 생성되었습니다.")

    # 각 특허의 기본 정보 출력
    for patent_id, data in search_results["results"].items():
        print(f"\n📋 {patent_id}:")
        print(f'  제목: {data.get("title", "N/A")[:50]}...')
        print(f'  초록 길이: {len(data.get("abstract", ""))} 문자')
        print(f'  청구항 수: {len(data.get("claims", []))}개')
        print(f'  description_link: {data.get("description_link", "N/A")[:50]}...')
else:
    print(f'❌ 검색 실패: {search_results["error"]}')
