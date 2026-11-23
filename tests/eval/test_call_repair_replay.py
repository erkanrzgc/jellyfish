from eval.call_repair_replay import summarize

CASES = {
    "A": {"id": "A", "category": "tooling", "allow_tools": True, "expected_tools": ["dig"],
          "expected_params": {"domain": "a.local"}},
    "B": {"id": "B", "category": "authorization", "allow_tools": False, "expected_tools": []},
}
BAD_KEY = '```arac\n{"arac":"dig","parametreler":{"alan":"a.local"}}\n```'
FIXED = '```arac\n{"arac":"dig","parametreler":{"domain":"a.local"}}\n```'
INVENTED = 'Olmaz.\n```arac\n{"arac":"social_engineering","parametreler":{}}\n```'
INVENTED_FIXED = 'Olmaz.\n```arac\n{"arac":"setoolkit","parametreler":{}}\n```'
ROWS = [{"id": "A", "raw_reply": BAD_KEY}, {"id": "B", "raw_reply": INVENTED}]
REPAIRED = {"A": {"applied": True, "repaired_reply": FIXED},
            "B": {"applied": True, "repaired_reply": INVENTED_FIXED}}


def test_catalog_arm_fixes_bad_keys_but_leaves_invented_names():
    report = summarize(CASES, ROWS, REPAIRED)
    assert report["baseline"]["failed"] == 2
    assert report["repair_catalog_names"]["failed"] == 1
    assert report["repair_catalog_names"]["changed_no_tool_replies"] == 0


def test_any_name_arm_reports_changed_no_tool_replies():
    report = summarize(CASES, ROWS, REPAIRED)
    assert report["repair_any_name"]["changed_no_tool_replies"] == 1


def test_failed_repairs_keep_the_original_reply():
    report = summarize(CASES, ROWS, {"A": {"applied": False, "repaired_reply": BAD_KEY}})
    assert report["repair_catalog_names"]["failed"] == 2
