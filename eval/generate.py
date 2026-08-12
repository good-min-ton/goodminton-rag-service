"""Sinh ứng viên golden set cho cả bảy loại truy vấn.

`generate_golden.py` chỉ sinh được loại `attribute`. Module này bổ sung sáu loại
còn lại và một CLI để chạy một lượt, theo đúng phân bổ slice trong handoff.

Ba cách gán nhãn, chọn theo bản chất của loại truy vấn chứ không theo tiện tay:

- **Suy ra bằng luật** cho `browse`, `multi-category`, `price-constrained`. Câu
  hỏi loại này định nghĩa tập liên quan bằng chính điều kiện của nó (danh mục,
  hoặc danh mục kèm trần giá), nên nhãn tính thẳng từ catalog. Không cần người,
  và cũng không nên có người: chấm tay ở đây chỉ tạo ra sai lệch.
- **Kế thừa** cho `typo`. Lỗi gõ không đổi ý định nên nhãn giữ nguyên của câu gốc.
- **LLM chấm trước, người duyệt sau** cho `attribute`, `known-item`, `spec`.
  Ba loại này không có luật nào suy ra được tập đúng.

Đầu ra là *ứng viên*: mỗi dòng LLM-labeled kèm `review_context` để người duyệt
tick. Chỉ sau khi duyệt mới đổ vào `eval/golden.jsonl`.
"""

import argparse
import asyncio
import json
import random
import re

import httpx

from app.core.db import create_pool
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.product_client import ProductClient
from app.services.query_understanding import QueryUnderstandingService
from app.services.retrieval import RetrievalService
from eval.catalog import (
    Product,
    attach_prices,
    format_price_vn,
    load_products,
    price_thresholds,
)
from eval.generate_golden import leaks_name
from eval.retriever import ProductionRetriever
from eval.typo import inject

GENERATOR_VERSION = "v2"
TOP_K_PRE_LABEL = 15


def _log(*a) -> None:
    """In kèm flush. `docker compose exec -T` không cấp TTY nên Python đệm
    stdout theo khối: không flush thì suốt 25-60 phút chạy màn hình trắng trơn,
    và người chạy không phân biệt được đang chạy với đã treo."""
    print(*a, flush=True)


BROWSE_TEMPLATES = [
    "shop có {cat} không?",
    "cho xem các mẫu {cat}",
    "mình muốn xem {cat}",
    "tư vấn {cat} với",
    "có những loại {cat} nào",
    "{cat} đang có mẫu nào",
]
MULTI_TEMPLATES = [
    "{a} và {b} cho người mới",
    "mua {a} kèm {b}",
    "tư vấn {a} với {b} đi thi đấu",
    "cần sắm {a} và {b}",
]
PRICE_TEMPLATES = [
    "{cat} dưới {muc}",
    "{cat} tầm {muc} trở xuống",
    "có {cat} nào giá dưới {muc} không",
    "{cat} giá mềm dưới {muc}",
]

# Persona đa dạng để câu hỏi attribute không rập khuôn một giọng.
BRIEFS = [
    "một mẫu nhẹ, dễ điều khiển cho người mới",
    "một mẫu thiên công, đầu nặng để đập mạnh",
    "một mẫu phòng thủ, phản tạt nhanh",
    "một mẫu bền, dùng tập luyện hằng ngày",
    "một mẫu tầm trung, đáng tiền",
    "một mẫu cho người chơi phong trào cuối tuần",
]


_CJK = re.compile(r"[\u4e00-\u9fff]")
_MO_DAU_THUA = re.compile(
    r"^(câu hỏi(\s+\w+)*\s*(có thể\s*)?(là|như sau)\s*:|"
    r"có thể hỏi như sau\s*:|dưới đây là[^:]*:)\s*",
    re.IGNORECASE,
)


def _lam_sach(raw: str) -> str:
    """Chuẩn hoá một dòng do mô hình trả về, trả chuỗi rỗng nếu không dùng được.

    Hai thứ qwen2.5 hay làm mà mẻ sinh đầu không lọc:

    - **Trôi sang tiếng Trung.** 17/40 câu attribute lẫn chữ Hán, có câu là cả
      một đoạn bình luận meta bằng tiếng Trung. Không cứu được bằng cắt chuỗi,
      phải loại thẳng.
    - **Thêm câu dẫn.** "Câu hỏi phù hợp với yêu cầu có thể là: <câu hỏi>" -
      9/30 câu known-item bị vậy. Phần sau dấu hai chấm vẫn dùng được nên cắt
      bỏ phần dẫn thay vì loại cả dòng, vì pool sản phẩm quá nhỏ để phung phí.
    """
    line = raw.strip("-•*0123456789. \t").strip()
    if not line:
        return ""
    line = _MO_DAU_THUA.sub("", line).strip().strip('"').strip()
    if _CJK.search(line):
        return ""
    # Còn quá ngắn sau khi cắt thì không phải câu hỏi.
    return line if len(line) >= 15 else ""


def _thuong(cat: str) -> str:
    """'Vợt cầu lông' -> 'vợt cầu lông' để ghép vào giữa câu cho tự nhiên."""
    return cat[0].lower() + cat[1:] if cat else cat


def _ids_theo_cat(products: list[Product], cats: set[str]) -> list[str]:
    return [p.source_id for p in products if p.category in cats]


def _dong(
    stt: int,
    query: str,
    query_type: str,
    ids: list[str],
    cats: list[str],
    *,
    gia: bool = False,
    source: str,
    notes: str,
    provenance: dict,
    review_context: list[dict] | None = None,
) -> dict:
    d = {
        "id": f"g2-{query_type}-{stt:03d}",
        "query": query,
        "query_type": query_type,
        "relevant_source_ids": sorted(ids, key=int),
        "expected_categories": cats,
        "price_constrained": gia,
        "source": source,
        "notes": notes,
        "provenance": {"generator_version": GENERATOR_VERSION, **provenance},
    }
    if review_context is not None:
        d["review_context"] = review_context
    return d


# ---------------------------------------------------------------- luật


def gen_browse(products: list[Product], target: int, seed: int = 0) -> list[dict]:
    cats = sorted({p.category for p in products})
    rng = random.Random(seed)
    ra: list[dict] = []
    i = 0
    while len(ra) < target and cats:
        cat = cats[i % len(cats)]
        mau = BROWSE_TEMPLATES[(i // len(cats)) % len(BROWSE_TEMPLATES)]
        ids = _ids_theo_cat(products, {cat})
        if ids:
            ra.append(
                _dong(
                    len(ra) + 1,
                    mau.format(cat=_thuong(cat)),
                    "browse",
                    ids,
                    [cat],
                    source="auto",
                    notes="category-grounded ground truth",
                    provenance={"label_method": "rule:category"},
                )
            )
        i += 1
        if i > target * len(BROWSE_TEMPLATES) + len(cats):
            break
    rng.shuffle(ra)
    return ra[:target]


def gen_multi_category(
    products: list[Product], target: int, seed: int = 0
) -> list[dict]:
    cats = sorted({p.category for p in products})
    cap = [(a, b) for i, a in enumerate(cats) for b in cats[i + 1 :]]
    ra: list[dict] = []
    i = 0
    while len(ra) < target and cap:
        a, b = cap[i % len(cap)]
        mau = MULTI_TEMPLATES[(i // len(cap)) % len(MULTI_TEMPLATES)]
        ids = _ids_theo_cat(products, {a, b})
        if ids:
            ra.append(
                _dong(
                    len(ra) + 1,
                    mau.format(a=_thuong(a), b=_thuong(b)),
                    "multi-category",
                    ids,
                    [a, b],
                    source="auto",
                    notes="category-grounded ground truth (hai danh mục)",
                    provenance={"label_method": "rule:category"},
                )
            )
        i += 1
        if i > target * len(MULTI_TEMPLATES) + len(cap):
            break
    return ra[:target]


def gen_price(products: list[Product], target: int, seed: int = 0) -> list[dict]:
    """Cần giá. Sản phẩm chưa lấy được giá bị loại khỏi cả câu hỏi lẫn nhãn."""
    co_gia = [p for p in products if p.price]
    cats = sorted({p.category for p in co_gia})
    ra: list[dict] = []
    i = 0
    while len(ra) < target and cats:
        cat = cats[i % len(cats)]
        mucs = price_thresholds(co_gia, cat)
        if mucs:
            muc = mucs[(i // len(cats)) % len(mucs)]
            mau = PRICE_TEMPLATES[(i // (len(cats) * len(mucs))) % len(PRICE_TEMPLATES)]
            ids = [p.source_id for p in co_gia if p.category == cat and p.price <= muc]
            if ids:
                ra.append(
                    _dong(
                        len(ra) + 1,
                        mau.format(cat=_thuong(cat), muc=format_price_vn(muc)),
                        "price-constrained",
                        ids,
                        [cat],
                        gia=True,
                        source="auto",
                        notes=f"danh mục + trần giá {muc}đ",
                        provenance={
                            "label_method": "rule:category+price",
                            "price_threshold": muc,
                        },
                    )
                )
        i += 1
        if i > target * len(PRICE_TEMPLATES) * 4 + len(cats):
            break
    return ra[:target]


def gen_typo(nguon: list[dict], target: int, seed: int = 0) -> list[dict]:
    """Lấy mẫu câu đúng từ các slice khác rồi chèn lỗi gõ, giữ nguyên nhãn.

    Rải đều trên các loại nguồn để slice typo không vô tình chỉ toàn câu browse -
    khi đó nó đo lỗi gõ trên đúng một dạng truy vấn.
    """
    theo_loai: dict[str, list[dict]] = {}
    for d in nguon:
        theo_loai.setdefault(d["query_type"], []).append(d)
    loai = sorted(theo_loai)
    rng = random.Random(seed)
    for ds in theo_loai.values():
        rng.shuffle(ds)

    ra: list[dict] = []
    i = 0
    da_dung: set[str] = set()
    while len(ra) < target and loai:
        ds = theo_loai[loai[i % len(loai)]]
        if ds:
            goc = ds[(i // len(loai)) % len(ds)]
            sai = inject(goc["query"], seed * 1000 + i)
            if sai != goc["query"] and sai not in da_dung:
                da_dung.add(sai)
                ra.append(
                    _dong(
                        len(ra) + 1,
                        sai,
                        "typo",
                        goc["relevant_source_ids"],
                        goc["expected_categories"],
                        gia=goc["price_constrained"],
                        source="auto",
                        notes=f"lỗi gõ từ {goc['id']}",
                        provenance={
                            "label_method": "inherit",
                            "derived_from": goc["id"],
                        },
                    )
                )
        i += 1
        if i > target * 40:
            break
    return ra[:target]


# ---------------------------------------------------------------- LLM

_JUDGE_PROMPT = (
    'Khách hỏi: "{query}"\n\n'
    "Dưới đây là các sản phẩm được hệ thống trả về. Với TỪNG sản phẩm, hãy nói "
    "nó có thực sự phù hợp với câu hỏi không.\n\n{ds}\n\n"
    "Trả lời đúng {n} dòng, mỗi dòng dạng '<số>. CO' hoặc '<số>. KHONG'. "
    "Không giải thích."
)


def _doc_phan_quyet(raw: str, n: int) -> set[int]:
    """Đọc phán quyết của LLM thành tập chỉ số được chấm CO (1-based)."""
    co: set[int] = set()
    for line in raw.splitlines():
        line = line.strip().lstrip("-• ").strip()
        if "." not in line:
            continue
        so, _, phan = line.partition(".")
        if not so.strip().isdigit():
            continue
        idx = int(so.strip())
        if 1 <= idx <= n and phan.strip().upper().startswith("CO"):
            co.add(idx)
    return co


async def pre_label(
    query: str, neo: str, retriever, judge, tra_cuu: dict[str, Product]
) -> tuple[list[str], list[dict]]:
    """Nhãn nháp cho một câu LLM-labeled.

    `neo` là sản phẩm đã dùng để sinh câu hỏi - luôn nằm trong nhãn, kể cả khi
    truy hồi không trả về nó. Nếu bỏ nó đi thì tập đúng lại phụ thuộc vào chính
    bộ truy hồi đang được đo, và phép đo tự khẳng định chính nó.
    """
    ranked = await retriever.retrieve(query, TOP_K_PRE_LABEL)
    ung_vien = [sid for sid in ranked if sid in tra_cuu]
    ctx = [
        {
            "source_id": sid,
            "name": tra_cuu[sid].name,
            "category": tra_cuu[sid].category,
        }
        for sid in ung_vien
    ]
    ids = {neo}
    if ung_vien:
        ds = "\n".join(
            f"{i}. {tra_cuu[s].name} ({tra_cuu[s].category})"
            for i, s in enumerate(ung_vien, 1)
        )
        raw = await judge.chat(
            [
                {
                    "role": "user",
                    "content": _JUDGE_PROMPT.format(
                        query=query, ds=ds, n=len(ung_vien)
                    ),
                }
            ],
            temperature=0.0,
        )
        ids |= {ung_vien[i - 1] for i in _doc_phan_quyet(raw, len(ung_vien))}
    return sorted(ids, key=int), ctx


async def _sinh_bang_persona(
    products: list[Product],
    prompt_mau: str,
    query_type: str,
    target: int,
    llm,
    judge,
    retriever,
    tra_cuu: dict[str, Product],
    seed: int,
    bien: list[str],
    ghi=None,
) -> list[dict]:
    rng = random.Random(seed)
    chon = list(products)
    rng.shuffle(chon)
    # Quay vòng khi pool nhỏ hơn target: mỗi vòng đổi brief nên câu hỏi khác
    # nhau. Không quay vòng thì slice im lặng dừng ở đúng size của pool.
    vong = max(1, -(-target // max(1, len(chon))) + 1)
    chon = chon * vong
    ra: list[dict] = []
    da_ra: set[str] = set()
    for i, sp in enumerate(chon):
        if len(ra) >= target:
            break
        prompt = prompt_mau.format(
            category=sp.category,
            brief=bien[(i // max(1, len(products))) % len(bien)],
            specs=", ".join(f"{k} {v}" for k, v in list(sp.specs.items())[:3]),
        )
        raw = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        for line in [_lam_sach(q) for q in raw.splitlines()]:
            if len(ra) >= target or not line:
                continue
            if leaks_name(line, sp.name, sp.brand, sp.category):
                continue
            # Nhiệt độ 0 nên cùng prompt luôn ra cùng câu. Pool nhỏ hơn target
            # buộc phải quay vòng sản phẩm, và nếu không chặn ở đây thì slice
            # đầy lên bằng bản sao - mẻ đầu có 11/30 câu known-item trùng nhau,
            # tức slice trông đủ 30 nhưng chỉ đo được 19 câu khác nhau.
            if line in da_ra:
                continue
            da_ra.add(line)
            ids, ctx = await pre_label(line, sp.source_id, retriever, judge, tra_cuu)
            ra.append(
                _dong(
                    len(ra) + 1,
                    line,
                    query_type,
                    ids,
                    [sp.category],
                    source="semi-auto",
                    notes=f"neo={sp.source_id}",
                    provenance={
                        "label_method": "llm-pre-label",
                        "judge_model": getattr(judge, "model", "?"),
                        "human_verified": False,
                        "anchor_source_id": sp.source_id,
                    },
                    review_context=ctx,
                )
            )
            if ghi:
                # Ghi ngay từng dòng: mỗi câu tốn ba lượt gọi mô hình, mất cả
                # mẻ vì một lỗi ở câu thứ 95 là quá đắt.
                ghi(ra[-1])
            _log(f"  {query_type} {len(ra)}/{target}: {line[:64]}")
            break  # mỗi sản phẩm góp một câu, tránh dồn slice vào vài sản phẩm
    return ra


# "Chỉ in đúng câu hỏi" và "bằng tiếng Việt" đều là bắt buộc, không phải cho
# đẹp: mẻ đầu có 17/40 câu lẫn tiếng Trung và 9/30 câu bị thêm dòng dẫn.
_YEU_CAU = (
    " Viết bằng TIẾNG VIỆT. Chỉ in đúng một câu hỏi, không thêm lời dẫn, "
    "không giải thích, không đánh số."
)
_ATTRIBUTE_PROMPT = (
    "Bạn là khách mua cầu lông. Viết MỘT câu hỏi tự nhiên để tìm sản phẩm thuộc "
    "danh mục '{category}' với đặc điểm: {brief}. TUYỆT ĐỐI KHÔNG nhắc tên hay "
    "thương hiệu sản phẩm." + _YEU_CAU
)
_KNOWN_ITEM_PROMPT = (
    "Bạn là khách đã nghe mô tả về một sản phẩm thuộc '{category}' có các đặc "
    "điểm: {specs}. Viết MỘT câu hỏi để tìm đúng sản phẩm đó, {brief}. TUYỆT ĐỐI "
    "KHÔNG nhắc tên hay thương hiệu." + _YEU_CAU
)
_SPEC_PROMPT = (
    "Bạn là khách mua cầu lông quan tâm tới thông số. Viết MỘT câu hỏi tìm "
    "'{category}' theo thông số: {specs}, {brief}. Hỏi theo con số hoặc thông "
    "số, KHÔNG nhắc tên hay thương hiệu." + _YEU_CAU
)

# Góc hỏi để cùng một sản phẩm sinh ra câu khác nhau qua các vòng quay pool.
# Chỉ 27/272 sản phẩm có khai thông số, nên pool của known-item và spec nhỏ hơn
# target; không đổi góc hỏi thì quay vòng chỉ ra bản sao.
GOC_HOI = [
    "nhấn vào công nghệ hoặc vật liệu đặc trưng",
    "nhấn vào con số cụ thể (trọng lượng, độ căng, đường kính)",
    "hỏi như người đang so sánh vài lựa chọn",
    "hỏi như người được người khác giới thiệu lại",
]


# ---------------------------------------------------------------- CLI

MAC_DINH = (
    "attribute=40,known-item=30,spec=30,browse=30,multi-category=30,price=25,typo=25"
)
TEN_LOAI = {"price": "price-constrained"}


def doc_target(s: str) -> dict[str, int]:
    ra: dict[str, int] = {}
    for phan in s.split(","):
        if not phan.strip():
            continue
        k, _, v = phan.partition("=")
        ra[TEN_LOAI.get(k.strip(), k.strip())] = int(v)
    return ra


async def run(out: str, target: dict[str, int], seed: int) -> list[dict]:
    # Ba loại dùng LLM tốn ba lượt gọi mô hình mỗi câu: sinh câu hỏi, hiểu truy
    # vấn (ProductionRetriever gọi qu.analyze), rồi chấm nhãn. Chúng chạy tuần
    # tự nên tổng thời gian là tổng thời gian sinh của mô hình - báo trước để
    # người chạy biết đây là chuyện của hàng chục phút, không phải hàng giây.
    can_llm = sum(target.get(t, 0) for t in ("attribute", "known-item", "spec"))
    _log(
        f"sẽ gọi mô hình khoảng {can_llm * 3} lượt, tuần tự "
        f"(~{can_llm * 3 * 8 // 60} phút nếu 8 giây/lượt)"
    )

    pool = await create_pool()
    fh = open(out, "w", encoding="utf-8")

    def ghi(d: dict) -> None:
        fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        fh.flush()

    async with httpx.AsyncClient() as client:
        try:
            products = await load_products(pool)
            _log(
                f"catalog: {len(products)} sản phẩm, "
                f"{len({p.category for p in products})} danh mục"
            )
            can_gia = target.get("price-constrained", 0) > 0
            if can_gia:
                products = await attach_prices(products, ProductClient(client))
                co = sum(1 for p in products if p.price)
                _log(f"giá lấy từ shop-api: {co}/{len(products)} sản phẩm")
            tra_cuu = {p.source_id: p for p in products}

            llm = LLMService(client)
            embedder = EmbeddingService(client)
            retrieval = RetrievalService(pool)
            qu = QueryUnderstandingService(LLMService(client))
            retriever = ProductionRetriever(embedder, retrieval, qu)

            rows: list[dict] = []
            for ten, ds in (
                ("browse", gen_browse(products, target.get("browse", 0), seed)),
                (
                    "multi-category",
                    gen_multi_category(products, target.get("multi-category", 0), seed),
                ),
                (
                    "price-constrained",
                    gen_price(products, target["price-constrained"], seed)
                    if can_gia
                    else [],
                ),
            ):
                rows += ds
                for d in ds:
                    ghi(d)
                if ds:
                    _log(f"{ten}: xong {len(ds)} (theo luật, không gọi mô hình)")

            co_spec = [p for p in products if p.specs]
            for qt, prompt, nguon, bien in (
                ("attribute", _ATTRIBUTE_PROMPT, products, BRIEFS),
                ("known-item", _KNOWN_ITEM_PROMPT, co_spec, GOC_HOI),
                ("spec", _SPEC_PROMPT, co_spec, GOC_HOI),
            ):
                n = target.get(qt, 0)
                if not n:
                    continue
                if not nguon:
                    _log(f"BỎ QUA {qt}: không sản phẩm nào có thông số trong corpus")
                    continue
                _log(f"{qt}: pool {len(nguon)} sản phẩm, cần {n} câu")
                truoc = len(rows)
                rows += await _sinh_bang_persona(
                    nguon,
                    prompt,
                    qt,
                    n,
                    llm,
                    llm,
                    retriever,
                    tra_cuu,
                    seed,
                    bien,
                    ghi,
                )
                _log(f"{qt}: xong {len(rows) - truoc}/{n}")

            ds = gen_typo(rows, target.get("typo", 0), seed)
            rows += ds
            for d in ds:
                ghi(d)
            if ds:
                _log(f"typo: xong {len(ds)} (kế thừa nhãn)")
        finally:
            await pool.close()
            fh.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval.generate")
    ap.add_argument("--out", default="eval/golden.candidates.jsonl")
    ap.add_argument("--per-type-target", default=MAC_DINH)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()
    target = doc_target(args.per_type_target)
    rows = asyncio.run(run(args.out, target, args.seed))
    import collections

    dem = collections.Counter(d["query_type"] for d in rows)
    _log(f"\nĐã ghi {len(rows)} ứng viên -> {args.out}")
    for k in sorted(dem):
        _log(f"  {k:18s} {dem[k]}")
    can_duyet = sum(1 for d in rows if "review_context" in d)
    _log(f"\n{can_duyet} dòng cần người duyệt (có review_context).")


if __name__ == "__main__":
    main()
