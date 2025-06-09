import json

print("🔍 샘플 특허 데이터 생성")
print("=" * 40)

# 샘플 특허 데이터 (실제 구조와 유사하게)
sample_data = {
    "US20210390793A1": {
        "title": "Advanced Machine Learning System for Patent Analysis and Classification",
        "abstract": "A comprehensive machine learning system that automatically analyzes patent documents, extracts key technical features, and classifies patents according to their technological domains. The system uses natural language processing, deep learning models, and knowledge graphs to provide accurate patent analysis.",
        "claims": [
            "A method for processing patent data comprising: collecting patent documents from multiple sources; extracting technical features using natural language processing; training machine learning models on the extracted features; and classifying patents based on technological domains.",
            "The method of claim 1, wherein the natural language processing includes tokenization, named entity recognition, and semantic analysis of patent text.",
            "The method of claim 1, further comprising generating knowledge graphs from patent relationships and citation networks.",
            "A system implementing the method of claim 1, comprising: a data collection module; a feature extraction module; a machine learning module; and a classification module.",
            "The system of claim 4, wherein the machine learning module uses deep neural networks for patent classification.",
        ],
        "description": "This invention relates to patent analysis systems. In recent years, the volume of patent applications has grown exponentially, making manual analysis increasingly difficult.",
        "description_link": "https://serpapi.com/searches/sample1/google_patents_details/description.html",
        "filing_date": "2021-01-27",
        "publication_date": "2023-08-22",
        "application_date": "2021-01-27",
        "status": "Published",
        "inventor": ["John Smith", "Jane Doe"],
        "assignee": ["Tech Innovation Corp"],
        "classifications": [
            {"code": "G06F", "description": "Electric digital data processing"}
        ],
        "citations": {"cited_by": [], "cites": []},
        "google_patents_url": "https://patents.google.com/patent/US20210390793A1/en",
        "search_timestamp": "2025-06-08T23:00:00",
    },
    "US11630280B2": {
        "title": "Distributed Computing Architecture for Real-time Data Processing",
        "abstract": "A distributed computing system designed for processing large-scale real-time data streams. The architecture includes multiple processing nodes, load balancing mechanisms, and fault tolerance features to ensure high availability and performance.",
        "claims": [
            "A distributed computing system comprising: multiple processing nodes configured in a cluster; a load balancer for distributing data processing tasks; and a fault tolerance mechanism for handling node failures.",
            "The system of claim 1, wherein the processing nodes use parallel processing algorithms for real-time data analysis.",
            "The system of claim 1, further comprising a data storage layer for persistent data management.",
            "A method for processing real-time data streams using the system of claim 1, comprising: receiving data streams; distributing processing tasks; and aggregating results.",
        ],
        "description": "Background: Real-time data processing has become crucial for modern applications. Existing systems often face scalability and reliability challenges.",
        "description_link": "https://serpapi.com/searches/sample2/google_patents_details/description.html",
        "filing_date": "2020-05-15",
        "publication_date": "2022-11-10",
        "application_date": "2020-05-15",
        "status": "Granted",
        "inventor": ["Alice Johnson", "Bob Wilson"],
        "assignee": ["Data Systems Inc"],
        "classifications": [
            {"code": "H04L", "description": "Transmission of digital information"}
        ],
        "citations": {"cited_by": [], "cites": []},
        "google_patents_url": "https://patents.google.com/patent/US11630280B2/en",
        "search_timestamp": "2025-06-08T23:00:00",
    },
    "US11630282B2": {
        "title": "Quantum-Enhanced Cryptographic Security Protocol",
        "abstract": "A novel cryptographic protocol that leverages quantum computing principles to enhance security in digital communications. The protocol uses quantum key distribution and entanglement to provide unprecedented levels of encryption security.",
        "claims": [
            "A cryptographic method comprising: generating quantum keys using quantum entanglement; distributing the keys through quantum channels; and encrypting data using the quantum-generated keys.",
            "The method of claim 1, wherein quantum entanglement is maintained throughout the key distribution process.",
            "The method of claim 1, further comprising detecting eavesdropping attempts through quantum state monitoring.",
            "A quantum cryptographic system implementing the method of claim 1, comprising: a quantum key generator; quantum communication channels; and encryption/decryption modules.",
        ],
        "description": "Field of Invention: This invention relates to quantum cryptography and secure communications. The increasing threat of quantum computers to traditional encryption methods necessitates new approaches.",
        "description_link": "https://serpapi.com/searches/sample3/google_patents_details/description.html",
        "filing_date": "2019-12-03",
        "publication_date": "2022-11-15",
        "application_date": "2019-12-03",
        "status": "Granted",
        "inventor": ["Dr. Carol Chen", "Dr. David Martinez"],
        "assignee": ["Quantum Security Solutions"],
        "classifications": [
            {"code": "H04L", "description": "Transmission of digital information"}
        ],
        "citations": {"cited_by": [], "cites": []},
        "google_patents_url": "https://patents.google.com/patent/US11630282B2/en",
        "search_timestamp": "2025-06-08T23:00:00",
    },
}

# 캐시 파일로 저장
with open("patent_cache.json", "w", encoding="utf-8") as f:
    json.dump(sample_data, f, ensure_ascii=False, indent=2)

print("✅ patent_cache.json 샘플 데이터가 생성되었습니다.")

# 데이터 요약 출력
for patent_id, data in sample_data.items():
    print(f"\n📋 {patent_id}:")
    print(f'  제목: {data["title"][:50]}...')
    print(f'  초록 길이: {len(data["abstract"])} 문자')
    print(f'  청구항 수: {len(data["claims"])}개')
    print(f'  발명자: {", ".join(data["inventor"])}')
    print(f'  출원일: {data["filing_date"]}')
