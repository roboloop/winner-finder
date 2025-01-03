import bisect
from typing import List


class CircleSectors:
    def __init__(self, sectors: List[tuple[float, float]]):
        self._sectors = sorted(sectors, key=lambda x: x[0])
        self._lot_names = [{} for _ in self._sectors]
        self._starts = [start for start, _ in self._sectors]
        self._votes = [0 for _ in self._sectors]

    def _get_sector_index(self, angle: float) -> int:
        idx = bisect.bisect_right(self._starts, angle)

        return idx - 1 if idx > 0 else len(self._starts) - 1

    def add_lot_name(self, angle: float, lot_name: str) -> None:
        idx = self._get_sector_index(angle)
        lot_names = self._lot_names[idx]
        lot_names[lot_name] = lot_names[lot_name] + 1 if lot_name in lot_names else 1

    def vote(self, angle: float) -> None:
        idx = self._get_sector_index(angle)
        self._votes[idx] += 1

    def _format_lot_name(self, idx: int) -> str:
        lot_names = self._lot_names[idx]
        if not len(lot_names):
            return f"lot name #{idx}"

        total = sum(lot_names.values())
        percentages = [(lot_name, (votes / total) * 100) for lot_name, votes in lot_names.items()]
        sorted_percentages = sorted(percentages, key=lambda x: x[1], reverse=True)

        return " | ".join([f"{lot_name}" for lot_name, percentage in sorted_percentages])

    def _get_sector_stat(self, idx: int) -> tuple[float, float, str, float]:
        votes_percent = (self._votes[idx] / sum(self._votes)) * 100
        synthetic_percent = 0.0
        if len(self._lot_names[idx]) == 1:
            lot_name = next(iter(self._lot_names[idx]))
            for index, obj in enumerate(self._lot_names):
                if index == idx:
                    continue
                if lot_name in obj:
                    synthetic_percent += (self._votes[index] / sum(self._votes)) * 100 / len(obj)

        start, end = self._sectors[idx]
        lot_percent = (360.0 + end - start) % 360 / 360.0 * 100

        return round(votes_percent, 2), round(synthetic_percent, 2), self._format_lot_name(idx), round(lot_percent, 2)

    def most_voted(self) -> List[tuple[float, float, str, float]]:
        top = 3
        non_zero_votes = [(i, v) for i, v in enumerate(self._votes) if v > 0]
        idxs = [i for i, _ in sorted(non_zero_votes, key=lambda x: x[1], reverse=True)[:top]]

        return [self._get_sector_stat(idx) for idx in idxs]

    def by_angle(self, angle: float) -> tuple[float, float, str, float]:
        idx = self._get_sector_index(angle)

        return self._get_sector_stat(idx)
