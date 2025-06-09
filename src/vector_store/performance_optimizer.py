#!/usr/bin/env python3
"""Vector Store 성능 최적화 모듈"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""

    operation: str
    duration: float
    timestamp: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """성능 모니터링 클래스"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        logger.info(f"성능 모니터 초기화: 최대 {max_history}개 메트릭 보관")

    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """작업 성능 기록"""
        metric = PerformanceMetrics(
            operation=operation,
            duration=duration,
            timestamp=time.time(),
            success=success,
            metadata=metadata or {},
        )

        self.metrics_history.append(metric)

        if success:
            self.operation_stats[operation].append(duration)
            if len(self.operation_stats[operation]) > 100:
                self.operation_stats[operation] = self.operation_stats[operation][-100:]

    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """특정 작업의 통계 정보 반환"""
        durations = self.operation_stats.get(operation, [])

        if not durations:
            return {
                "count": 0,
                "avg_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
            }

        return {
            "count": len(durations),
            "avg_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
        }

    def get_overall_stats(self) -> Dict[str, Any]:
        """전체 성능 통계 반환"""
        total_operations = len(self.metrics_history)
        successful_operations = sum(1 for m in self.metrics_history if m.success)

        if total_operations == 0:
            return {
                "total_operations": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "operations_by_type": {},
            }

        success_rate = successful_operations / total_operations
        avg_duration = statistics.mean([m.duration for m in self.metrics_history])

        operations_by_type = {}
        for operation in self.operation_stats:
            operations_by_type[operation] = self.get_operation_stats(operation)

        return {
            "total_operations": total_operations,
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "operations_by_type": operations_by_type,
        }

    def identify_bottlenecks(self) -> List[str]:
        """성능 병목 지점 식별"""
        bottlenecks = []

        for operation, durations in self.operation_stats.items():
            if len(durations) < 5:
                continue

            avg_duration = statistics.mean(durations)
            if avg_duration > 2.0:  # 2초 이상이면 병목으로 간주
                bottlenecks.append(f"{operation} (평균: {avg_duration:.3f}초)")

        return bottlenecks

    def generate_recommendations(self) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        stats = self.get_overall_stats()

        # 성공률 기반 권장사항
        if stats["success_rate"] < 0.95:
            recommendations.append(
                f"성공률이 {stats['success_rate']:.1%}로 낮습니다. 오류 처리 및 재시도 로직을 개선하세요."
            )

        # 평균 응답 시간 기반 권장사항
        if stats["avg_duration"] > 2.0:
            recommendations.append(
                f"평균 응답 시간이 {stats['avg_duration']:.2f}초로 높습니다. 배치 크기 조정을 고려하세요."
            )

        # 작업별 권장사항
        for operation, op_stats in stats["operations_by_type"].items():
            if op_stats["avg_duration"] > 3.0:
                recommendations.append(
                    f"{operation} 작업이 평균 {op_stats['avg_duration']:.2f}초로 느립니다. "
                    "캐싱이나 인덱싱을 고려하세요."
                )

        # 병목 기반 권장사항
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            recommendations.append(
                f"다음 작업들이 병목으로 식별되었습니다: {', '.join(bottlenecks)}. "
                "우선적으로 최적화하세요."
            )

        if not recommendations:
            recommendations.append(
                "현재 성능이 양호합니다. 지속적인 모니터링을 권장합니다."
            )

        return recommendations


def performance_monitor(monitor: PerformanceMonitor, operation_name: str):
    """성능 모니터링 데코레이터"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            result = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                logger.error(f"{operation_name} 실행 실패: {e}")
                raise
            finally:
                duration = time.time() - start_time
                metadata = {"args_count": len(args), "kwargs_count": len(kwargs)}

                if result is not None and hasattr(result, "__len__"):
                    metadata["result_size"] = len(result)

                monitor.record_operation(operation_name, duration, success, metadata)

        return wrapper

    return decorator
