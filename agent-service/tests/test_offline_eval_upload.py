"""rag_eval.upload 离线评测分数回写测试：trace_id解析与上传行为"""

from rag_eval.upload import SCORE_COMMENT, trace_id_from_row, upload_scores


class FakeClient:
    def __init__(self, fail: bool = False):
        self.scores: list[dict] = []
        self.fail = fail

    def create_score(self, **kwargs):
        if self.fail:
            raise RuntimeError("upload failed")
        self.scores.append(kwargs)


class TestTraceIdFromRow:
    def test_normal_value(self):
        assert trace_id_from_row({"trace_id": "trace-abc"}) == "trace-abc"

    def test_strips_whitespace(self):
        assert trace_id_from_row({"trace_id": "  trace-abc  "}) == "trace-abc"

    def test_empty_and_missing(self):
        assert trace_id_from_row({"trace_id": ""}) is None
        assert trace_id_from_row({"trace_id": None}) is None
        assert trace_id_from_row({}) is None

    def test_pandas_nan_and_placeholder_texts(self):
        # 旧答题卡重新加载后空列会变成NaN，不能当成合法trace_id
        assert trace_id_from_row({"trace_id": float("nan")}) is None
        assert trace_id_from_row({"trace_id": "nan"}) is None
        assert trace_id_from_row({"trace_id": "None"}) is None


class TestUploadScores:
    def test_uploads_each_metric(self):
        client = FakeClient()
        uploaded = upload_scores(
            {"trace_id": "trace-abc"},
            {"faithfulness": 0.9, "answer_relevancy": 0.8},
            client=client,
        )

        assert uploaded == 2
        assert [s["name"] for s in client.scores] == ["faithfulness", "answer_relevancy"]
        for score in client.scores:
            assert score["trace_id"] == "trace-abc"
            assert score["data_type"] == "NUMERIC"
            assert score["comment"] == SCORE_COMMENT

    def test_skips_when_trace_id_missing(self):
        client = FakeClient()
        uploaded = upload_scores({"trace_id": ""}, {"faithfulness": 0.9}, client=client)
        assert uploaded == 0
        assert client.scores == []

    def test_empty_scores_returns_zero(self):
        client = FakeClient()
        assert upload_scores({"trace_id": "t"}, {}, client=client) == 0
        assert client.scores == []

    def test_upload_failure_does_not_raise(self):
        client = FakeClient(fail=True)
        # 单条失败只记日志，不影响判卷主流程
        assert upload_scores({"trace_id": "t"}, {"faithfulness": 0.9}, client=client) == 0
