#!/usr/bin/env python3
import json
import os


def find_description_samples():
    cache_file = "src/cache/patent_cache.json"

    if not os.path.exists(cache_file):
        print(f"캐시 파일이 없습니다: {cache_file}")
        return

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
    except Exception as e:
        print(f"캐시 파일 로드 오류: {e}")
        return

    print(f"총 캐시 항목 수: {len(cache_data)}")

    samples_with_description = []

    # 모든 캐시 항목을 순회하며 설명이 있는 것 찾기
    for key, item in cache_data.items():
        if "data" not in item:
            continue

        data = item["data"]

        # 여러 가능한 설명 필드들 체크
        description_candidates = []

        # 1. 직접적인 description 필드
        if (
            "description" in data
            and isinstance(data["description"], str)
            and len(data["description"]) > 0
        ):
            description_candidates.append(("description", len(data["description"])))

        # 2. scraped_description 필드
        if (
            "scraped_description" in data
            and isinstance(data["scraped_description"], str)
            and len(data["scraped_description"]) > 0
        ):
            description_candidates.append(
                ("scraped_description", len(data["scraped_description"]))
            )

        # 3. scraping_results 안의 description
        if "scraping_results" in data and isinstance(data["scraping_results"], dict):
            scraping_results = data["scraping_results"]
            if (
                "description" in scraping_results
                and isinstance(scraping_results["description"], str)
                and len(scraping_results["description"]) > 0
            ):
                description_candidates.append(
                    (
                        "scraping_results.description",
                        len(scraping_results["description"]),
                    )
                )

        # 4. 다른 중첩 구조들 체크
        for field_name, field_value in data.items():
            if isinstance(field_value, dict):
                if (
                    "description" in field_value
                    and isinstance(field_value["description"], str)
                    and len(field_value["description"]) > 0
                ):
                    description_candidates.append(
                        (f"{field_name}.description", len(field_value["description"]))
                    )

        # 가장 긴 설명을 가진 필드 선택
        if description_candidates:
            best_desc = max(description_candidates, key=lambda x: x[1])
            samples_with_description.append((key, best_desc[0], best_desc[1]))

    print(f"\n설명이 있는 샘플 수: {len(samples_with_description)}")

    if samples_with_description:
        # 길이순으로 정렬
        samples_with_description.sort(key=lambda x: x[2], reverse=True)

        print("\n상위 5개 샘플:")
        for i, (key, field, length) in enumerate(samples_with_description[:5]):
            print(f"  {i+1}. 키: {key[:30]}... | 필드: {field} | 길이: {length:,} 문자")

        # 가장 긴 샘플의 내용 미리보기
        best_key, best_field, best_length = samples_with_description[0]
        print(f"\n가장 긴 설명 샘플 미리보기 ({best_length:,} 문자):")
        print("=" * 50)

        # 실제 내용 가져오기
        item = cache_data[best_key]
        data = item["data"]

        if "." in best_field:
            field_parts = best_field.split(".")
            content = data
            for part in field_parts:
                content = content[part]
        else:
            content = data[best_field]

        print(content[:300])
        if len(content) > 300:
            print(f"\n... (총 {len(content):,} 문자 중 300자만 표시)")

        return best_key, best_field
    else:
        print("설명이 있는 샘플을 찾지 못했습니다.")
        return None, None


if __name__ == "__main__":
    find_description_samples()
