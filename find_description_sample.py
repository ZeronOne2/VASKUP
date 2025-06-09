#!/usr/bin/env python3
"""캐시에서 description이 있는 샘플 찾기"""

import json


def find_samples_with_description():
    """캐시에서 description이 있는 샘플들을 찾습니다."""

    with open("src/cache/patent_cache.json", "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    print(f"총 캐시 항목 수: {len(cache_data)}")

    samples_with_description = []

    for key, item in cache_data.items():
        if "data" not in item:
            continue

        data = item["data"]

        # description 필드 확인
        description = data.get("description", "")
        if isinstance(description, str) and len(description) > 0:
            # 특허번호 추출
            patent_number = None
            if "raw_data" in data and "publication_number" in data["raw_data"]:
                patent_number = data["raw_data"]["publication_number"]
            elif "google_patents_url" in data:
                import re

                match = re.search(r"/patent/([^/]+)/", data["google_patents_url"])
                if match:
                    patent_number = match.group(1)

            if patent_number:
                samples_with_description.append(
                    (
                        key,
                        patent_number,
                        len(description),
                        data.get("title", "Unknown"),
                        description[:100],
                    )
                )

    print(f"\n설명이 있는 샘플 수: {len(samples_with_description)}")

    if samples_with_description:
        # 길이순으로 정렬
        samples_with_description.sort(key=lambda x: x[2], reverse=True)

        print("\n설명이 있는 샘플들:")
        for i, (key, patent_num, desc_length, title, preview) in enumerate(
            samples_with_description
        ):
            print(f"  {i+1}. {patent_num} ({desc_length:,} 문자)")
            print(f"     제목: {title}")
            print(f"     미리보기: {preview}...")
            print()

        # 가장 긴 설명을 가진 샘플 반환
        best_sample = samples_with_description[0]
        return best_sample[0], best_sample[1]  # key, patent_number
    else:
        print("설명이 있는 샘플을 찾지 못했습니다.")
        return None, None


if __name__ == "__main__":
    find_samples_with_description()
