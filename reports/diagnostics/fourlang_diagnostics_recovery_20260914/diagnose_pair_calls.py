"""Non-training local-model comparison on a small, authored diagnostic set."""
from __future__ import annotations

import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get('FOURLANG_DIAGNOSTIC_PROJECT', Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))

# Authored diagnostic references, not a certified/native-reviewed benchmark.
CASES = [
    ("会议九点开始。", "Uchrashuv soat to'qqizda boshlanadi.", "Встреча начинается в девять часов."),
    ("我今天不去学校。", "Men bugun maktabga bormayman.", "Я сегодня не пойду в школу."),
    ("请给我一杯水。", "Iltimos, menga bir stakan suv bering.", "Пожалуйста, дайте мне стакан воды."),
    ("这本书多少钱？", "Bu kitob qancha turadi?", "Сколько стоит эта книга?"),
    ("我买了三个苹果。", "Men uchta olma sotib oldim.", "Я купил три яблока."),
    ("他昨天给我打了电话。", "U kecha menga telefon qildi.", "Он вчера мне позвонил."),
    ("我们明天早上出发。", "Biz ertaga ertalab yo'lga chiqamiz.", "Мы отправимся завтра утром."),
    ("请不要打开窗户。", "Iltimos, derazani ochmang.", "Пожалуйста, не открывайте окно."),
    ("医院在银行旁边。", "Shifoxona bank yonida joylashgan.", "Больница находится рядом с банком."),
    ("我把钥匙忘在家里了。", "Men kalitni uyda unutib qoldirdim.", "Я забыл ключ дома."),
    ("如果下雨，我们就不去公园。", "Agar yomg'ir yog'sa, biz bog'ga bormaymiz.", "Если пойдёт дождь, мы не пойдём в парк."),
    ("我想买一张去塔什干的车票。", "Men Toshkentga chipta sotib olmoqchiman.", "Я хочу купить билет до Ташкента."),
    ("火车下午3点到达。", "Poyezd tushdan keyin soat 3 da yetib keladi.", "Поезд прибудет в 3 часа дня."),
    ("这个房间没有热水。", "Bu xonada issiq suv yo'q.", "В этой комнате нет горячей воды."),
    ("请把地址发给我。", "Iltimos, manzilni menga yuboring.", "Пожалуйста, отправьте мне адрес."),
    ("我已经吃过早饭了。", "Men allaqachon nonushta qildim.", "Я уже позавтракал."),
    ("他还没有回来。", "U hali qaytib kelmadi.", "Он ещё не вернулся."),
    ("这不是我的手机。", "Bu mening telefonim emas.", "Это не мой телефон."),
    ("请慢一点说。", "Iltimos, sekinroq gapiring.", "Пожалуйста, говорите медленнее."),
    ("我听不懂这句话。", "Men bu gapni tushunmayapman.", "Я не понимаю это предложение."),
    ("我们需要等十分钟。", "Biz o'n daqiqa kutishimiz kerak.", "Нам нужно подождать десять минут."),
    ("商店晚上八点关门。", "Do'kon kechqurun soat sakkizda yopiladi.", "Магазин закрывается в восемь часов вечера."),
    ("请向左转，然后一直走。", "Iltimos, chapga buriling, keyin to'g'ri yuring.", "Пожалуйста, поверните налево, затем идите прямо."),
    ("孩子正在睡觉。", "Bola uxlayapti.", "Ребёнок спит."),
    ("我每天坐公交车上班。", "Men har kuni ishga avtobusda boraman.", "Я каждый день езжу на работу на автобусе."),
    ("这件衣服太大了。", "Bu kiyim juda katta.", "Эта одежда слишком большая."),
    ("我只需要两张票。", "Menga faqat ikkita chipta kerak.", "Мне нужны только два билета."),
    ("请先洗手，再吃饭。", "Iltimos, avval qo'lingizni yuving, keyin ovqatlaning.", "Пожалуйста, сначала вымойте руки, потом ешьте."),
    ("因为道路封闭，公交车晚点了。", "Yo'l yopilgani uchun avtobus kechikdi.", "Автобус опоздал, потому что дорога была закрыта."),
    ("这家商店周日不开门。", "Bu do'kon yakshanba kuni ochilmaydi.", "Этот магазин не работает по воскресеньям."),
]


def main():
    import torch
    from inference.engine import TranslationEngine
    from inference.loader import load_translation_model
    from scripts.pipeline_v2.seq2seq_flow import prepare_inputs, translate
    from src.model_utils import load_tokenizer

    out = Path(os.environ.get('FOURLANG_DIAGNOSTIC_OUTPUT', ROOT / 'reports/diagnostics/pair_call_comparison')) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    cases = [dict(zip(('zh', 'uz', 'ru'), row)) for row in CASES]
    (out / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = json.loads((ROOT / 'configs/specialists/current_pair_models.json').read_text())
    config = {'training': {'max_source_length': 256}, 'deployment': {'num_beams': 5, 'max_new_tokens': 256}}
    report = {'purpose': 'diagnostic_only_not_testset_selection', 'references': 'authored_not_native_certified', 'config': config, 'models': []}
    for pair in ('zh_uz', 'uz_ru'):
        current = next(p['model_path'] for p in manifest['pairs'] if p['id'] == pair)
        for version, relative in [('current', current), ('baseline_exp1', f'results/student/pair_specialists/{pair}/exp1/best_model/shared')]:
            path = ROOT / relative
            loaded = load_translation_model(path, device='cuda', dtype='float16')
            eval_tokenizer = load_tokenizer(str(path), 'small100')
            row = {'pair': pair, 'version': version, 'path': str(path), 'model_class': type(loaded.model).__name__, 'generation_config': loaded.model.generation_config.to_dict(), 'directions': []}
            for source, target in [pair.split('_'), pair.split('_')[::-1]]:
                texts = [c[source] for c in cases]
                engine = TranslationEngine(loaded, direction=f'{source}-{target}', num_beams=5, max_source_length=256, max_new_tokens=256)
                token_matches = []
                for text in texts:
                    a, _ = prepare_inputs(eval_tokenizer, 'small100', source, target, [text], 256)
                    loaded.tokenizer.tgt_lang = target
                    b = loaded.tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=256)
                    token_matches.append(all(torch.equal(a[k], b[k]) for k in a))
                predictions = translate(eval_tokenizer, loaded.model, 'small100', source, target, texts, config)
                direct = [engine.translate(t)['translation'] for t in texts]
                result = {'direction': f'{source}-{target}', 'token_matches': sum(token_matches), 'rows': [
                    {'id': i + 1, 'source': t, 'reference': cases[i][target], 'evaluation_call': predictions[i], 'direct_call': direct[i]}
                    for i, t in enumerate(texts)]}
                row['directions'].append(result)
                (out / f'{pair}_{version}_{source}-{target}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'{pair} {version} {source}-{target}: token_match={sum(token_matches)}/30', flush=True)
            report['models'].append(row)
            (out / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            del engine, eval_tokenizer, loaded
            gc.collect()
            torch.cuda.empty_cache()
    print('FINISHED ' + str(out), flush=True)


if __name__ == '__main__':
    main()
