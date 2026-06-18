from src.metrics.base_metric import BaseMetric


class SpeedMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, batch_time=None, **kwargs):
        if batch_time is None:
            return None
        return float(batch_time)
