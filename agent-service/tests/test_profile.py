"""app.agents.profile 用户画像测试：Store读写、不可变合并与摘要"""

import asyncio
from types import SimpleNamespace

from app.agents.profile import (
    UserProfile,
    delete_profile,
    get_profile,
    merge_profile,
    save_profile,
    summarize_profile,
)


class FakeStore:
    """内存版Store：命名空间+键作为唯一标识，模拟真实的aget/aput/adelete语义"""

    def __init__(self):
        self.data: dict[tuple, dict] = {}
        self.writes: list[tuple] = []

    async def aget(self, namespace: tuple, key: str):
        value = self.data.get((namespace, key))
        return None if value is None else SimpleNamespace(value=value)

    async def aput(self, namespace: tuple, key: str, value: dict):
        self.data[(namespace, key)] = value
        self.writes.append((namespace, key))

    async def adelete(self, namespace: tuple, key: str):
        self.data.pop((namespace, key), None)


class TestMergeProfile:
    def test_none_fields_do_not_override(self):
        current = UserProfile(age=30, budget="8000元")
        merged = merge_profile(current, UserProfile(family="已婚"))
        assert merged.age == 30
        assert merged.budget == "8000元"
        assert merged.family == "已婚"

    def test_original_object_not_mutated(self):
        current = UserProfile(age=30)
        merge_profile(current, UserProfile(age=35))
        # 不可变合并：原对象保持原值
        assert current.age == 30

    def test_empty_list_clears_field(self):
        current = UserProfile(existing_coverage=["医疗险"])
        merged = merge_profile(current, UserProfile(existing_coverage=[]))
        assert merged.existing_coverage == []


class TestStoreRoundTrip:
    def test_get_missing_profile_returns_none(self):
        assert asyncio.run(get_profile(FakeStore(), user_id=1)) is None

    def test_save_then_get_roundtrip(self):
        store = FakeStore()
        asyncio.run(save_profile(store, 1, UserProfile(age=30, budget="8000元")))
        profile = asyncio.run(get_profile(store, 1))
        assert profile is not None
        assert profile.age == 30
        assert profile.budget == "8000元"

    def test_second_save_merges_instead_of_overwrite(self):
        store = FakeStore()
        asyncio.run(save_profile(store, 1, UserProfile(age=30)))
        asyncio.run(save_profile(store, 1, UserProfile(family="已婚")))
        profile = asyncio.run(get_profile(store, 1))
        assert profile.age == 30
        assert profile.family == "已婚"

    def test_profiles_isolated_per_user(self):
        store = FakeStore()
        asyncio.run(save_profile(store, 1, UserProfile(age=30)))
        asyncio.run(save_profile(store, 2, UserProfile(age=45)))
        assert asyncio.run(get_profile(store, 1)).age == 30
        assert asyncio.run(get_profile(store, 2)).age == 45


class TestDeleteProfile:
    def test_deletes_existing_profile(self):
        store = FakeStore()
        asyncio.run(save_profile(store, 1, UserProfile(age=30)))

        assert asyncio.run(delete_profile(store, 1)) is True
        # 删除后读不到画像
        assert asyncio.run(get_profile(store, 1)) is None

    def test_missing_profile_returns_false(self):
        assert asyncio.run(delete_profile(FakeStore(), 1)) is False

    def test_only_target_user_deleted(self):
        store = FakeStore()
        asyncio.run(save_profile(store, 1, UserProfile(age=30)))
        asyncio.run(save_profile(store, 2, UserProfile(age=45)))

        asyncio.run(delete_profile(store, 1))

        assert asyncio.run(get_profile(store, 1)) is None
        # 其他用户画像不受影响
        assert asyncio.run(get_profile(store, 2)).age == 45


class TestSummarize:
    def test_all_fields_present(self):
        profile = UserProfile(
            age=30,
            family="已婚",
            budget="8000元",
            existing_coverage=["医疗险"],
            health_notes="高血压",
            preferences="消费型",
            notes="关注性价比",
        )
        assert summarize_profile(profile) == (
            "年龄：30；家庭：已婚；预算：8000元；已有保障：医疗险；"
            "健康备注：高血压；偏好：消费型；备注：关注性价比"
        )

    def test_partial_profile_skips_empty_fields(self):
        assert summarize_profile(UserProfile(age=30)) == "年龄：30"
