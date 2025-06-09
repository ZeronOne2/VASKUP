#!/usr/bin/env python3
"""캐시 데이터 구조 탐색 및 설명 필드 확인"""

import json


def explore_cache_structure():
    """캐시 데이터 구조를 탐색하여 설명 필드를 찾습니다."""

    with open("src/cache/patent_cache.json", "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    print(f"총 캐시 항목 수: {len(cache_data)}")
    print("\n각 항목의 필드 구조 확인:")

    samples_with_description = []

    for i, (key, item) in enumerate(cache_data.items()):
        if i >= 10:  # 처음 10개만 확인
            break

        print(f"\n=== 항목 {i+1} (키: {key[:20]}...) ===")
        print(f"최상위 필드: {list(item.keys())}")

        if "data" in item:
            data = item["data"]
            print(f"data 필드: {list(data.keys())}")

            # 설명 관련 필드들 확인
            desc_fields = ["description", "scraped_description", "scraping_results"]
            found_description = False

            for field in desc_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, str):
                        print(f"{field}: {len(value)} 문자")
                        if len(value) > 0:
                            print(f"  미리보기: {value[:100]}...")
                            found_description = True
                            if key not in samples_with_description:
                                samples_with_description.append(
                                    (key, field, len(value))
                                )
                    else:
                        print(f"{field}: {type(value)} 타입")
                        if isinstance(value, dict) and value:
                            print(f"  하위 필드: {list(value.keys())}")
                            # 딕셔너리 내부에서 설명 찾기
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, str) and len(sub_value) > 100:
                                    print(f"  {field}.{sub_key}: {len(sub_value)} 문자")
                                    if "description" in sub_key.lower():
                                        samples_with_description.append(
                                            (key, f"{field}.{sub_key}", len(sub_value))
                                        )

        # scraping_results나 다른 중첩 필드도 확인
        if "scraping_results" in item:
            print(f'scraping_results 필드 존재: {type(item["scraping_results"])}')

    print(f"\n\n📊 설명이 있는 샘플들:")
    if samples_with_description:
        for key, field, length in samples_with_description:
            print(f"  키: {key[:20]}... | 필드: {field} | 길이: {length:,} 문자")

        # 가장 긴 설명을 가진 샘플 반환
        best_sample = max(samples_with_description, key=lambda x: x[2])
        print(
            f"\n🎯 가장 긴 설명을 가진 샘플: {best_sample[0][:20]}... ({best_sample[2]:,} 문자)"
        )
        return best_sample[0], best_sample[1]
    else:
        print("  설명이 있는 샘플을 찾지 못했습니다.")
        return None, None


def show_sample_content(sample_key, field_path):
    """특정 샘플의 설명 내용을 보여줍니다."""

    with open("src/cache/patent_cache.json", "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    if sample_key not in cache_data:
        print(f"샘플 키 {sample_key}를 찾을 수 없습니다.")
        return None

    item = cache_data[sample_key]

    # 중첩 필드 경로 처리
    if "." in field_path:
        field_parts = field_path.split(".")
        content = item["data"]
        for part in field_parts:
            content = content[part]
    else:
        content = item["data"][field_path]

    print(f"\n📄 샘플 내용 ({len(content):,} 문자):")
    print("=" * 60)
    print(content[:500])
    if len(content) > 500:
        print(f"\n... (총 {len(content):,} 문자 중 500자만 표시)")
    print("=" * 60)

    return content


if __name__ == "__main__":
    print("🔍 캐시 데이터 구조 탐색 시작...\n")

    # 구조 탐색
    sample_key, field_path = explore_cache_structure()

    # 샘플 내용 표시
    if sample_key and field_path:
        content = show_sample_content(sample_key, field_path)

        print(f"\n✅ 설명이 있는 샘플을 찾았습니다!")
        print(f"   키: {sample_key}")
        print(f"   필드: {field_path}")
        print(f"   길이: {len(content):,} 문자")
    else:
        print("\n❌ 설명이 있는 샘플을 찾지 못했습니다.")
