"""Tests for the optional min_date filter threaded through train_models."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytz

from backend.learning import LearningEngine
from ml.train import train_models


def _mock_engine() -> MagicMock:
    engine = MagicMock(spec=LearningEngine)
    engine.timezone = pytz.timezone("Europe/Stockholm")
    engine.db_path = ":memory:"
    engine.config = {}
    return engine


class TestTrainMinDate:
    """train_models passes min_date into _load_slot_observations as start_time."""

    @patch("ml.train._load_slot_observations")
    @patch("ml.train.get_learning_engine")
    def test_default_min_date_is_no_op(self, mock_get_engine, mock_load):
        """Default min_date=None -> start_time=None (no lower-bound filter)."""
        mock_get_engine.return_value = _mock_engine()
        # Empty observations -> train_models returns early after loading.
        mock_load.return_value = pd.DataFrame()

        train_models()

        assert mock_load.call_count == 1
        assert mock_load.call_args.kwargs["start_time"] is None

    @patch("ml.train._load_slot_observations")
    @patch("ml.train.get_learning_engine")
    def test_min_date_passed_as_start_time(self, mock_get_engine, mock_load):
        """An explicit min_date is forwarded as the start_time filter."""
        mock_get_engine.return_value = _mock_engine()
        mock_load.return_value = pd.DataFrame()

        tz = pytz.timezone("Europe/Stockholm")
        min_date = datetime.now(tz) - timedelta(days=1)

        train_models(min_date=min_date)

        assert mock_load.call_count == 1
        assert mock_load.call_args.kwargs["start_time"] == min_date
