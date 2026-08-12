"""Ảnh chụp catalog dùng cho việc sinh golden set.

Giá KHÔNG nằm trong kb_chunks. `indexer.strip_pricing` cố tình xoá mọi số tiền
khỏi văn bản được index, còn `metadata` chỉ giữ đúng một khoá `category`. Đó là
lựa chọn đúng cho RAG - giá là dữ liệu sống, phải hỏi qua tool lúc trả lời chứ
không được đóng băng vào vector. Nhưng nó có nghĩa là loại truy vấn theo giá
không thể gán nhãn từ corpus, phải lấy giá từ shop-api.

Thông số kỹ thuật thì ngược lại: chúng nằm sẵn trong chunk đầu của mỗi sản phẩm
dưới dòng "Thông số: ...", nên đọc thẳng từ đó.
"""

import asyncio
from dataclasses import dataclass, replace

from eval.generate_golden import parse_name_brand

# Số lời gọi shop-api chạy song song khi lấy giá. Đủ nhanh cho 272 sản phẩm mà
# không dội một lúc mấy trăm kết nối vào backend.
SONG_SONG = 8


@dataclass(frozen=True)
class Product:
    source_id: str
    name: str
    brand: str
    category: str
    specs: dict[str, str]
    price: int | None = None


def parse_specs(content: str) -> dict[str, str]:
    """Đọc dòng "Thông số: tên: giá trị | tên: giá trị" ở chunk đầu.

    Trả về dict rỗng khi sản phẩm không khai thông số (indexer ghi "N/A").
    """
    for line in content.splitlines():
        if not line.startswith("Thông số: "):
            continue
        raw = line[len("Thông số: ") :].strip()
        if not raw or raw == "N/A":
            return {}
        out: dict[str, str] = {}
        for phan in raw.split(" | "):
            if ": " in phan:
                ten, gia_tri = phan.split(": ", 1)
                ten, gia_tri = ten.strip(), gia_tri.strip()
                if ten and gia_tri:
                    out[ten] = gia_tri
        return out
    return {}


async def load_products(pool) -> list[Product]:
    """Mỗi sản phẩm đúng một dòng: chunk_index=0 là chunk chứa header."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT source_id, metadata->>'category' AS category, content "
            "FROM kb_chunks WHERE doc_type='product' AND chunk_index=0 "
            "ORDER BY source_id"
        )
    out: list[Product] = []
    for r in rows:
        if not r["category"]:
            continue
        name, brand = parse_name_brand(r["content"])
        out.append(
            Product(
                source_id=str(r["source_id"]),
                name=name,
                brand=brand,
                category=r["category"],
                specs=parse_specs(r["content"]),
            )
        )
    return out


def _gia_thap_nhat(pricing: dict) -> int | None:
    """Giá đại diện của sản phẩm = biến thể rẻ nhất, tính theo giá đang bán.

    Lấy nhỏ nhất chứ không phải trung bình: khách hỏi "vợt dưới 1 triệu" là hỏi
    có mua được với 1 triệu không, chỉ cần một biến thể đạt là đủ.
    """
    gia = []
    for v in pricing.get("variants") or []:
        x = v.get("salePrice") or v.get("price")
        if isinstance(x, (int, float)) and x > 0:
            gia.append(int(x))
    return min(gia) if gia else None


async def attach_prices(products: list[Product], client) -> list[Product]:
    """Bổ sung giá từ shop-api. Sản phẩm lỗi/không giá giữ price=None."""
    sem = asyncio.Semaphore(SONG_SONG)

    async def mot(p: Product) -> Product:
        async with sem:
            try:
                pricing = await client.get_pricing(int(p.source_id))
            except Exception:
                return p
        return replace(p, price=_gia_thap_nhat(pricing))

    return list(await asyncio.gather(*(mot(p) for p in products)))


def format_price_vn(dong: int) -> str:
    """Số tiền viết như người mua nói: 500k, 1 triệu, 1,5 triệu."""
    if dong < 1_000_000:
        return f"{round(dong / 1_000)}k"
    trieu = dong / 1_000_000
    if abs(trieu - round(trieu)) < 0.05:
        return f"{round(trieu)} triệu"
    return f"{trieu:.1f}".replace(".", ",") + " triệu"


def _lam_tron(dong: int) -> int:
    """Bo về mốc giá người ta hay nói, để câu hỏi nghe tự nhiên."""
    if dong < 1_000_000:
        return max(100_000, round(dong / 100_000) * 100_000)
    return round(dong / 500_000) * 500_000


def price_thresholds(
    products: list[Product], category: str, phan_vi=(0.35, 0.6, 0.85)
) -> list[int]:
    """Ngưỡng giá lấy từ phân vị THẬT của danh mục.

    Không dùng mốc cố định: mỗi danh mục có dải giá riêng, một ngưỡng đặt tuỳ
    tiện sẽ cho tập liên quan rỗng hoặc bằng cả danh mục - cả hai đều làm câu
    hỏi mất giá trị đo lường.
    """
    gia = sorted(p.price for p in products if p.category == category and p.price)
    if not gia:
        return []
    out: list[int] = []
    for q in phan_vi:
        vt = min(len(gia) - 1, int(q * len(gia)))
        muc = _lam_tron(gia[vt])
        # Bỏ ngưỡng không chia được tập nào, và ngưỡng trùng nhau sau khi bo.
        if (
            muc not in out
            and any(g <= muc for g in gia)
            and not all(g <= muc for g in gia)
        ):
            out.append(muc)
    return out
