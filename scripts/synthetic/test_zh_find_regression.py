from scripts.synthetic.renderer_v2 import (
    load_verb_policies,
    render_zh_verb_v2,
)


def main():

    policies = load_verb_policies()

    cases = [
        {
            "tense": "present",
            "polarity": "pos",
            "expected": "找到了",
        },
        {
            "tense": "present",
            "polarity": "neg",
            "expected": "没找到",
        },
        {
            "tense": "future",
            "polarity": "pos",
            "expected": "会找到",
        },
        {
            "tense": "future",
            "polarity": "neg",
            "expected": "不会找到",
        },
    ]

    for case in cases:

        actual = render_zh_verb_v2(
            verb_id="FIND",
            original_surface="找到",
            tense=case["tense"],
            polarity=case["polarity"],
            policies=policies,
        )

        assert (
            actual
            == case["expected"]
        ), (
            f"FAILED: "
            f"tense={case['tense']} "
            f"polarity={case['polarity']} "
            f"expected={case['expected']} "
            f"actual={actual}"
        )

        print(
            "PASS:",
            case["tense"],
            case["polarity"],
            "->",
            actual,
        )

    print()
    print(
        "ZH FIND regression: "
        "4/4 PASS"
    )


if __name__ == "__main__":
    main()