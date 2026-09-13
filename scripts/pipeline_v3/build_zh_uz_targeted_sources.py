"""Build deterministic, evaluation-isolated Chinese sources for zh->uz repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = (
    "commerce",
    "medical_emergency",
    "travel_hotel",
    "numbers_time",
    "negation_logic",
    "complex_workflow",
)
PROTECTED_DEFAULTS = (
    "acceptance/zh_uz_direct_v1/gold.jsonl",
    "data/multilingual/fourlang/exp1/train.jsonl",
    "data/multilingual/fourlang/exp1/validation.jsonl",
    "data/multilingual/fourlang/exp2/train.jsonl",
    "data/multilingual/fourlang/exp2/validation.jsonl",
    "data/multilingual/fourlang/exp3_v2/train.jsonl",
    "data/multilingual/fourlang/exp3_v2/validation.jsonl",
    "data/distillation/zh_uz/v4/train.jsonl",
)

PRODUCTS = (
    ("咖啡", "杯"), ("绿茶", "杯"), ("果汁", "瓶"), ("矿泉水", "瓶"),
    ("苹果", "公斤"), ("橙子", "公斤"), ("葡萄", "公斤"), ("面包", "袋"),
    ("衬衫", "件"), ("外套", "件"), ("毛巾", "条"), ("电池", "盒"),
    ("充电器", "个"), ("耳机", "副"), ("文件夹", "个"), ("零件", "箱"),
)
MEDICINES = ("青霉素", "阿司匹林", "布洛芬", "抗生素", "止痛药", "退烧药")
SYMPTOMS = ("胸痛", "呼吸困难", "头晕", "持续发烧", "腹痛", "皮疹", "恶心", "伤口出血")
PLACES = ("机场", "火车站", "汽车站", "医院", "银行", "酒店", "市中心", "仓库")
WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
PEOPLE = ("客户", "司机", "医生", "经理", "老师", "维修人员", "仓库主管", "前台工作人员")
DOCUMENTS = ("合同", "发票", "申请表", "护照复印件", "检测报告", "付款凭证", "装箱单", "许可证")
PATIENTS = ("这位患者", "孩子", "老人", "李先生", "王女士", "一名孕妇", "一名成年患者", "一名六岁儿童")
EXAMS = ("血液检查", "腹部检查", "胃镜检查", "手术前检查", "体检", "血糖检查")
ONSETS = ("今天早上", "昨天下午", "昨天晚上", "两天前", "三天前", "本周一")


def normalized(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", text).casefold()
        if char.isalnum()
    )


def read_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def protected_texts(paths: list[Path]) -> tuple[set[str], dict[str, str]]:
    texts: set[str] = set()
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        hashes[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for row in read_jsonl(path):
            for prefix in ("source_zh", "src_text", "tgt_text"):
                text = row.get(prefix)
                if not isinstance(text, str) or not text.strip():
                    continue
                if prefix == "src_text" and row.get("src_lang") not in (None, "zh"):
                    continue
                if prefix == "tgt_text" and row.get("tgt_lang") not in (None, "zh"):
                    continue
                texts.add(normalized(text))
    return texts, hashes


def commerce(rng: random.Random):
    a, b = rng.sample(PRODUCTS, 2)
    qty, other = rng.randint(2, 20), rng.randint(1, 5)
    discount = rng.choice((5, 10, 12, 15, 20, 25))
    templates = (
        (f"我订的是{a[0]}，不是{b[0]}，请不要发错。", ["entity", "negation"], {"wanted": a[0], "rejected": b[0]}),
        (f"请给我{qty}{a[1]}{a[0]}，其中{other}{a[1]}单独包装。", ["quantity", "scope"], {"total": qty, "separate": other, "item": a[0]}),
        (f"这批{a[0]}打{discount}%折，但配送费不参加折扣。", ["percentage", "negation"], {"discount_percent": discount, "shipping_discounted": False}),
        (f"如果{a[0]}没有现货，就改成{b[0]}，数量仍然是{qty}{b[1]}。", ["condition", "entity", "quantity"], {"fallback": b[0], "quantity": qty}),
        (f"商品拆封前可以退货，拆封后只能换货，不能退款。", ["condition", "negation", "policy"], {"opened": "exchange_only", "refund_after_open": False}),
        (f"总价是{qty * 17000}苏姆，已经支付{qty * 5000}苏姆定金。", ["money", "quantity"], {"total_som": qty * 17000, "deposit_som": qty * 5000}),
        (f"请把{a[0]}的发票开成公司名称，不要开成个人姓名。", ["terminology", "negation"], {"document": "发票", "bill_to": "company"}),
        (f"我没有现金，请确认这里能不能刷卡。", ["payment", "negation"], {"cash": False, "asks_card": True}),
    )
    return rng.choice(templates)


def medical(rng: random.Random):
    medicine = rng.choice(MEDICINES)
    symptom, symptom2 = rng.sample(SYMPTOMS, 2)
    patient = rng.choice(PATIENTS)
    exam = rng.choice(EXAMS)
    onset = rng.choice(ONSETS)
    hours = rng.choice((4, 6, 8, 12))
    maximum = 24 // hours
    temperature = rng.choice((38.2, 38.5, 39.0, 39.2, 39.5, 40.0))
    templates = (
        (f"{patient}对{medicine}过敏，请不要使用含有这种成分的药。", ["medical", "allergy", "negation"], {"patient": patient, "allergen": medicine, "must_not_administer": True}),
        (f"如果{patient}出现{symptom}或者{symptom2}，请立即叫救护车。", ["medical", "condition", "emergency"], {"patient": patient, "symptoms": [symptom, symptom2], "action": "ambulance"}),
        (f"{patient}使用的{medicine}每{hours}小时服一片，一天最多{maximum}片，连续服用{rng.randint(2, 14)}天。", ["medical", "dosage", "number"], {"medicine": medicine, "interval_hours": hours, "daily_max": maximum}),
        (f"{patient}从{onset}开始发烧，最高体温是{temperature}摄氏度。", ["medical", "time", "decimal"], {"patient": patient, "onset": onset, "temperature_c": temperature}),
        (f"{exam}前需要空腹{hours}小时，只能喝水，不能吃东西。", ["medical", "duration", "negation"], {"exam": exam, "fasting_hours": hours, "water_allowed": True, "food_allowed": False}),
        (f"{patient}的伤口从{onset}起没有出血，但走路时疼痛明显加重。", ["medical", "negation", "contrast"], {"patient": patient, "onset": onset, "bleeding": False, "pain_when_walking": True}),
        (f"{medicine}要饭后服用；如果{patient}忘记一次，不要一次吃两片。", ["medical", "condition", "negation"], {"medicine": medicine, "patient": patient, "after_meal": True, "double_dose": False}),
        (f"{rng.choice(PLACES)}门口是急救通道，任何车辆都不能停放，请保持畅通。", ["emergency", "negation", "safety"], {"parking_allowed": False, "keep_clear": True}),
    )
    return rng.choice(templates)


def travel(rng: random.Random):
    origin, destination = rng.sample(PLACES, 2)
    line = rng.randint(1, 9)
    platform = rng.randint(1, 18)
    minutes = rng.choice((15, 20, 30, 40, 45, 60, 90))
    carriage = rng.randint(3, 16)
    templates = (
        (f"从{origin}去{destination}，请在下一站换乘{line}号线。", ["travel", "transfer", "number"], {"origin": origin, "destination": destination, "line": line}),
        (f"从{origin}开往{destination}的{rng.choice(('列车', '航班', '长途汽车'))}晚点约{minutes}分钟。", ["travel", "delay", "number"], {"origin": origin, "destination": destination, "delay_minutes": minutes}),
        (f"我的座位在{carriage}号车厢，不在{platform}号车厢。", ["travel", "number", "negation"], {"correct_carriage": carriage, "wrong_carriage": platform}),
        (f"前往{destination}的{rng.choice(('列车', '汽车', '机场大巴'))}从{platform}号站台出发。", ["travel", "platform", "number"], {"destination": destination, "platform": platform}),
        (f"从{origin}到{destination}只买了单程票，还没有买返程票。", ["travel", "ticket", "negation"], {"origin": origin, "destination": destination, "ticket": "one_way", "return_ticket": False}),
        (f"如果从{origin}到{destination}的航班取消，可以免费改签，也可以申请退款。", ["travel", "condition", "policy"], {"origin": origin, "destination": destination, "cancelled": True, "free_rebook": True, "refund": True}),
        (f"这条路正在维修，去{destination}需要绕行大约{minutes}分钟。", ["travel", "detour", "number"], {"destination": destination, "extra_minutes": minutes}),
        (f"请给我一间远离电梯的安静房间，住{rng.randint(2, 7)}晚。", ["hotel", "location", "number"], {"away_from_lift": True}),
    )
    return rng.choice(templates)


def numbers_time(rng: random.Random):
    weekday, weekday2 = rng.sample(WEEKDAYS, 2)
    hour = rng.randint(7, 20)
    minute = rng.choice((0, 15, 30, 45))
    amount = rng.randrange(120000, 5000001, 10000)
    paid = rng.randrange(10000, amount, 10000)
    percent = rng.choice((5, 10, 15, 20, 25, 30))
    boxes = rng.randrange(40, 501, 10)
    first = rng.randrange(10, boxes, 10)
    templates = (
        (f"预约改到{weekday}{hour:02d}:{minute:02d}，不是{weekday2}。", ["date", "time", "negation"], {"weekday": weekday, "time": f"{hour:02d}:{minute:02d}", "not_weekday": weekday2}),
        (f"预算不能超过{amount}苏姆，目前已经支付{paid}苏姆。", ["money", "upper_bound"], {"budget_max_som": amount, "paid_som": paid}),
        (f"这批货共有{boxes}箱，今天先发{first}箱，剩下的明天发。", ["quantity", "sequence"], {"total_boxes": boxes, "today_boxes": first, "remaining_boxes": boxes - first}),
        (f"商品优惠{percent}%，运费仍按原价计算。", ["percentage", "scope"], {"product_discount_percent": percent, "shipping_discount_percent": 0}),
        (f"会议原计划开{rng.choice((60, 90, 120))}分钟，实际只开了{rng.choice((30, 45, 50))}分钟。", ["duration", "contrast"], {}),
        (f"商店{hour:02d}:{minute:02d}开门，在此之前不办理业务。", ["time", "negation"], {"opening_time": f"{hour:02d}:{minute:02d}"}),
        (f"每位参加者需要{rng.randint(2, 6)}份材料，共有{rng.randint(8, 30)}位参加者。", ["quantity", "multiplication"], {}),
        (f"合同有效期为{rng.randint(6, 36)}个月，到期前{rng.randint(10, 60)}天可以申请续签。", ["duration", "deadline"], {}),
    )
    return rng.choice(templates)


def negation_logic(rng: random.Random):
    person = rng.choice(PEOPLE)
    document = rng.choice(DOCUMENTS)
    weekday = rng.choice(WEEKDAYS)
    templates = (
        (f"除非{person}在{weekday}以前明确同意，否则不要修改{document}。", ["unless", "negation", "entity"], {"condition": "explicit_consent", "deadline": weekday, "action_without_condition": False}),
        (f"不是所有{document}都审核完了，共有{rng.randint(10, 40)}份，目前只审核了其中{rng.randint(2, 8)}份。", ["scope", "negation", "quantity"], {}),
        (f"即使{person}今天回复，我们也来不及在{weekday}以前完成{document}。", ["concession", "deadline", "negation"], {"document": document, "still_cannot_finish": True}),
        (f"只要没有新的故障，设备就能在{weekday}恢复运行。", ["condition", "negation", "date"], {"condition": "no_new_failure"}),
        (f"我不是不同意{person}提出的方案，只是还需要{rng.randint(2, 14)}天考虑。", ["double_negation", "duration"], {"proposer": person, "rejects": False}),
        (f"{document}可以延期到{weekday}，也可以取消，但不能改成今天生效。", ["alternatives", "negation"], {"document": document, "delay": True, "cancel": True, "today": False}),
        (f"除了{person}以外，其他人都已经在{weekday}以前提交了{document}。", ["exception", "scope", "aspect"], {"exception": person, "deadline": weekday, "others_submitted": True}),
        (f"虽然从{rng.choice(PLACES)}送来的包装外面湿了，里面的{document}并没有损坏。", ["concession", "negation", "entity"], {"outside_wet": True, "inside_damaged": False}),
    )
    return rng.choice(templates)


def complex_workflow(rng: random.Random):
    p1, p2 = rng.sample(PEOPLE, 2)
    d1, d2 = rng.sample(DOCUMENTS, 2)
    weekday = rng.choice(WEEKDAYS)
    morning_hour = rng.randint(8, 11)
    cutoff_hour = rng.randint(12, 20)
    templates = (
        (f"先让{p1}检查{d1}，确认无误后交给{p2}签字，再把扫描件发给我。", ["sequence", "people", "document"], {"steps": [p1, p2, "send_scan"]}),
        (f"如果{p1}在{weekday}以前没有回复，就把{d1}的截止日期顺延一周。", ["condition", "deadline", "duration"], {"delay_days": 7}),
        (f"第一箱放{d1}，第二箱放{d2}，容易损坏的物品必须单独包装。", ["ordering", "entity", "safety"], {"box1": d1, "box2": d2, "fragile_separate": True}),
        (f"上午{morning_hour}点查看仓库，下午讨论价格，{d1}改天再签。", ["sequence", "time", "document"], {"morning": "warehouse", "afternoon": "price", "document_later": d1}),
        (f"先确认机器已经断电，然后打开后盖；检查结束前不要接通电源。", ["sequence", "safety", "negation"], {"power_before_open": False, "power_during_check": False}),
        (f"{p1}负责联系客户，{p2}负责核对数量，我负责准备{d1}。", ["role_assignment", "people"], {"roles": [p1, p2, "speaker"]}),
        (f"雨停后再出发；如果到{cutoff_hour}点还不停，就取消当天去{rng.choice(PLACES)}的行程。", ["condition", "time", "cancellation"], {"cancel_if_still_raining": True}),
        (f"收到{d1}以后先核对金额，不一致时不要签署{d2}，并立即联系{p1}。", ["sequence", "condition", "negation"], {"sign_if_mismatch": False, "contact": p1}),
    )
    return rng.choice(templates)


GENERATORS = {
    "commerce": commerce,
    "medical_emergency": medical,
    "travel_hotel": travel,
    "numbers_time": numbers_time,
    "negation_logic": negation_logic,
    "complex_workflow": complex_workflow,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-category", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument(
        "--output", default="data/targeted/zh_uz/v1/source_candidates.jsonl"
    )
    parser.add_argument(
        "--manifest", default="data/targeted/zh_uz/v1/source_manifest.json"
    )
    parser.add_argument("--protected", action="append", default=[])
    parser.add_argument(
        "--allow-missing-protected",
        action="store_true",
        help="Development only; production generation must check every default file.",
    )
    args = parser.parse_args()
    if args.rows_per_category < 100:
        raise ValueError("Use at least 100 rows per category")

    protected_paths = [PROJECT_ROOT / item for item in PROTECTED_DEFAULTS]
    protected_paths.extend(PROJECT_ROOT / item for item in args.protected)
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in protected_paths
        if not path.is_file()
    ]
    if missing and not args.allow_missing_protected:
        raise FileNotFoundError(
            "Required protected datasets are missing; refusing a partial overlap check: "
            + ", ".join(missing)
        )
    blocked, hashes = protected_texts(protected_paths)
    rng = random.Random(args.seed)
    rows = []
    selected = set()
    for category in CATEGORIES:
        attempts = 0
        while sum(row["category"] == category for row in rows) < args.rows_per_category:
            attempts += 1
            if attempts > args.rows_per_category * 500:
                raise RuntimeError(f"Could not generate enough unique {category} rows")
            source, tags, facts = GENERATORS[category](rng)
            key = normalized(source)
            if not key or key in blocked or key in selected:
                continue
            selected.add(key)
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
            rows.append(
                {
                    "candidate_id": f"zh-uz-targeted-v1-{digest}",
                    "src_lang": "zh",
                    "tgt_lang": "uz",
                    "src_text": source,
                    "category": category,
                    "challenge_tags": tags,
                    "structured_facts": facts,
                    "origin": "CONTROLLED_AI_AUTHORED_SOURCE_TEMPLATE",
                    "usage": "TRAINING_CANDIDATE_NOT_APPROVED",
                }
            )

    output = PROJECT_ROOT / args.output
    manifest = PROJECT_ROOT / args.manifest
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"Existing output differs: {output}; use a new version")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    counts = Counter(row["category"] for row in rows)
    try:
        output_name = str(output.relative_to(PROJECT_ROOT))
    except ValueError:
        output_name = str(output)
    report = {
        "schema_version": 1,
        "status": "TARGETED_SOURCES_READY_NOT_TRANSLATED_NOT_APPROVED",
        "rows": len(rows),
        "rows_by_category": dict(counts),
        "seed": args.seed,
        "output": output_name,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "protected_files_checked": hashes,
        "missing_protected_files": missing,
        "exact_protected_overlap": 0,
        "near_duplicate_check": "NOT_PERFORMED",
        "training_data_written": False,
    }
    manifest_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != manifest_text:
        raise RuntimeError(f"Existing manifest differs: {manifest}; use a new version")
    manifest.write_text(manifest_text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
